#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "globalbounding.hpp"
#include "morton.hpp"

namespace {

constexpr unsigned int threads_per_block = 256;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

unsigned int block_count(std::size_t n) {
    const std::size_t required = n / threads_per_block + (n % threads_per_block != 0);
    if (required > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("Morton kernel grid exceeds the CUDA launch limit");
    }
    return static_cast<unsigned int>(required);
}

class CudaStream {
public:
    CudaStream() {
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~CudaStream() noexcept {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept {
        return stream_;
    }

    void synchronize(const char* operation) {
        check_cuda(cudaStreamSynchronize(stream_), operation);
    }

    void destroy(const char* operation) {
        check_cuda(cudaStreamDestroy(stream_), operation);
        stream_ = nullptr;
    }

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace

namespace {

__global__ void normalise_kernel(const double* a, const double* centre,
                        const double* side, double* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (a[i] - (*centre - *side / 2.0)) / *side;
    }
}

__global__ void quantise_kernel(const double* normalised,
                                QuantisedCoordinate* quantised,
                                std::uint32_t grid_size,
                                std::uint32_t max_q, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const double scaled = normalised[i] * static_cast<double>(grid_size);
        const double clamped =
            ::fmin(::fmax(scaled, 0.0), static_cast<double>(max_q));
        quantised[i] = static_cast<QuantisedCoordinate>(clamped);
    }
}

__device__ __forceinline__ std::uint32_t expand_bits(QuantisedCoordinate value){
    std::uint32_t x = value & 0x000003FFu;

    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x <<  8)) & 0x0300F00Fu;
    x = (x | (x <<  4)) & 0x030C30C3u;
    x = (x | (x <<  2)) & 0x09249249u;

    return x;
}

__global__ void interleave_bits(const QuantisedCoordinate* x, const QuantisedCoordinate* y, const QuantisedCoordinate* z, MortonKey* keys, std::size_t n){
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    keys[i] = (expand_bits(x[i]) << 2) |
                (expand_bits(y[i]) << 1) |
                expand_bits(z[i]);
}

} // namespace

MortonKeys::MortonKeys(const double* x, const double* y, const double* z,
                       std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument(
            "Cannot compute Morton keys for an empty particle set");
    }
    if (x == nullptr || y == nullptr || z == nullptr) {
        throw std::invalid_argument(
            "Morton coordinate device pointers must not be null");
    }
    if (count > std::numeric_limits<ParticleIndex>::max()) {
        throw std::overflow_error(
            "Particle count exceeds the Morton index representation");
    }

    keys_.resize(count);
    indices_.resize(count);
    thrust::sequence(indices_.begin(), indices_.end(), ParticleIndex{0});

    Bbox box(x, y, z, count);
    const double* box_values = box.device_data();
    thrust::device_vector<double> out_x(count);
    thrust::device_vector<double> out_y(count);
    thrust::device_vector<double> out_z(count);

    constexpr int bits = 10;
    constexpr std::uint32_t grid_size = 1u << bits;
    constexpr std::uint32_t max_q = grid_size - 1;

    thrust::device_vector<QuantisedCoordinate> q_x(count);
    thrust::device_vector<QuantisedCoordinate> q_y(count);
    thrust::device_vector<QuantisedCoordinate> q_z(count);

    normalise(x, y, z, box_values, box_values + 1, box_values + 2,
              box_values + 3, thrust::raw_pointer_cast(out_x.data()),
              thrust::raw_pointer_cast(out_y.data()),
              thrust::raw_pointer_cast(out_z.data()), count);
    quantise(thrust::raw_pointer_cast(out_x.data()),
             thrust::raw_pointer_cast(out_y.data()),
             thrust::raw_pointer_cast(out_z.data()),
             thrust::raw_pointer_cast(q_x.data()),
             thrust::raw_pointer_cast(q_y.data()),
             thrust::raw_pointer_cast(q_z.data()), grid_size, max_q, count);
    create_keys(thrust::raw_pointer_cast(q_x.data()),
                thrust::raw_pointer_cast(q_y.data()),
                thrust::raw_pointer_cast(q_z.data()), count);

    // Keep each original particle index paired with its key. Stability makes
    // equal-key ordering deterministic and matches radix-sort semantics.
    thrust::stable_sort_by_key(keys_.begin(), keys_.end(), indices_.begin());
}

void MortonKeys::normalise(const double* x, const double* y, const double* z,
        const double* centre_x, const double* centre_y, const double* centre_z,
        const double* side,
        double* out_x, double* out_y, double* out_z,
        std::size_t n){
    if (n == 0) {
        return;
    }

    const unsigned int blocks = block_count(n);
    std::array<CudaStream, dimensions> streams;

    normalise_kernel<<<blocks, threads_per_block, 0, streams[0].get()>>>(x, centre_x, side, out_x, n);
    check_cuda(cudaGetLastError(), "normalise x kernel launch");
    normalise_kernel<<<blocks, threads_per_block, 0, streams[1].get()>>>(y, centre_y, side, out_y, n);
    check_cuda(cudaGetLastError(), "normalise y kernel launch");
    normalise_kernel<<<blocks, threads_per_block, 0, streams[2].get()>>>(z, centre_z, side, out_z, n);
    check_cuda(cudaGetLastError(), "normalise z kernel launch");

    streams[0].synchronize("normalise x kernel execution");
    streams[1].synchronize("normalise y kernel execution");
    streams[2].synchronize("normalise z kernel execution");

    streams[0].destroy("destroy normalise x stream");
    streams[1].destroy("destroy normalise y stream");
    streams[2].destroy("destroy normalise z stream");
}

void MortonKeys::quantise(const double* out_x, const double* out_y, const double* out_z,
        QuantisedCoordinate* q_x, QuantisedCoordinate* q_y, QuantisedCoordinate* q_z,
        std::uint32_t grid_size, std::uint32_t max_q, std::size_t n) {
    if (n == 0) {
        return;
    }

    const unsigned int blocks = block_count(n);
    std::array<CudaStream, dimensions> streams;

    quantise_kernel<<<blocks, threads_per_block, 0, streams[0].get()>>>(out_x, q_x, grid_size, max_q, n);
    check_cuda(cudaGetLastError(), "quantise x kernel launch");
    quantise_kernel<<<blocks, threads_per_block, 0, streams[1].get()>>>(out_y, q_y, grid_size, max_q, n);
    check_cuda(cudaGetLastError(), "quantise y kernel launch");
    quantise_kernel<<<blocks, threads_per_block, 0, streams[2].get()>>>(out_z, q_z, grid_size, max_q, n);
    check_cuda(cudaGetLastError(), "quantise z kernel launch");

    streams[0].synchronize("quantise x kernel execution");
    streams[1].synchronize("quantise y kernel execution");
    streams[2].synchronize("quantise z kernel execution");

    streams[0].destroy("destroy quantise x stream");
    streams[1].destroy("destroy quantise y stream");
    streams[2].destroy("destroy quantise z stream");
}

void MortonKeys::create_keys(const QuantisedCoordinate* q_x, const QuantisedCoordinate* q_y, const QuantisedCoordinate* q_z, std::size_t n){
    if (n == 0) {
        return;
    }

    const unsigned int blocks = block_count(n);
    interleave_bits<<<blocks, threads_per_block>>>(q_x, q_y, q_z, thrust::raw_pointer_cast(keys_.data()), n);
    check_cuda(cudaGetLastError(), "interleave bits kernel launch");
    check_cuda(cudaDeviceSynchronize(), "interleave bits kernel execution");
}

const MortonKey *MortonKeys::keys_device_data() const noexcept {
    return thrust::raw_pointer_cast(keys_.data());
}

MortonKey *MortonKeys::keys_device_data() noexcept {
    return thrust::raw_pointer_cast(keys_.data());
}

const ParticleIndex *MortonKeys::indices_device_data() const noexcept {
    return thrust::raw_pointer_cast(indices_.data());
}

ParticleIndex *MortonKeys::indices_device_data() noexcept {
    return thrust::raw_pointer_cast(indices_.data());
}

std::size_t MortonKeys::size() const noexcept {
    return keys_.size();
}
