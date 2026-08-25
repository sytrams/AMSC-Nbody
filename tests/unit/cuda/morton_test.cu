#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cuda_test.hpp"
#include "morton.hpp"

namespace {

class DeviceCoordinates {
public:
    DeviceCoordinates(const std::vector<double>& x,
                      const std::vector<double>& y,
                      const std::vector<double>& z)
        : x_(x.begin(), x.end()), y_(y.begin(), y.end()),
          z_(z.begin(), z.end()) {
        if (x.empty() || x.size() != y.size() || x.size() != z.size()) {
            throw std::invalid_argument(
                "Test coordinates must have equal non-zero sizes");
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return x_.size(); }

    [[nodiscard]] const double* x() const noexcept {
        return thrust::raw_pointer_cast(x_.data());
    }

    [[nodiscard]] const double* y() const noexcept {
        return thrust::raw_pointer_cast(y_.data());
    }

    [[nodiscard]] const double* z() const noexcept {
        return thrust::raw_pointer_cast(z_.data());
    }

private:
    thrust::device_vector<double> x_;
    thrust::device_vector<double> y_;
    thrust::device_vector<double> z_;
};

template <typename T>
std::vector<T> copy_from_device(const T* source, std::size_t count) {
    std::vector<T> result(count);
    const cudaError_t status = cudaMemcpy(result.data(), source,
                                          count * sizeof(T),
                                          cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return result;
}

class MortonKeysTest : public CudaTest {};

TEST_F(MortonKeysTest, SortsKeysAndCarriesOriginalParticleIndices) {
    const DeviceCoordinates coordinates({3.0, -3.0, 1.0, -1.0},
                                        {3.0, -3.0, 1.0, -1.0},
                                        {3.0, -3.0, 1.0, -1.0});

    const MortonKeys morton(coordinates.x(), coordinates.y(), coordinates.z(),
                            coordinates.size());
    const auto keys = copy_from_device(morton.keys_device_data(), morton.size());
    const auto indices =
        copy_from_device(morton.indices_device_data(), morton.size());

    EXPECT_EQ(morton.size(), coordinates.size());
    EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
    EXPECT_EQ(indices, (std::vector<ParticleIndex>{1u, 3u, 2u, 0u}));
}

TEST_F(MortonKeysTest, ProducesDeterministicKeysForCoincidentParticles) {
    const DeviceCoordinates coordinates({2.0, 2.0, 2.0},
                                        {-4.0, -4.0, -4.0},
                                        {8.0, 8.0, 8.0});

    const MortonKeys morton(coordinates.x(), coordinates.y(), coordinates.z(),
                            coordinates.size());
    const auto keys = copy_from_device(morton.keys_device_data(), morton.size());
    const auto indices =
        copy_from_device(morton.indices_device_data(), morton.size());

    EXPECT_EQ(keys, (std::vector<MortonKey>{0x38000000u, 0x38000000u,
                                            0x38000000u}));
    EXPECT_EQ(indices, (std::vector<ParticleIndex>{0u, 1u, 2u}));
}

TEST(MortonKeysValidationTest, RejectsEmptyInputAndNullPointers) {
    EXPECT_THROW(MortonKeys(nullptr, nullptr, nullptr, 0),
                 std::invalid_argument);

    const auto* non_null =
        reinterpret_cast<const double*>(static_cast<std::uintptr_t>(1));
    EXPECT_THROW(MortonKeys(nullptr, non_null, non_null, 1),
                 std::invalid_argument);
    EXPECT_THROW(MortonKeys(non_null, nullptr, non_null, 1),
                 std::invalid_argument);
    EXPECT_THROW(MortonKeys(non_null, non_null, nullptr, 1),
                 std::invalid_argument);
}

} // namespace
