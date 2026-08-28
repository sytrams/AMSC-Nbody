#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>


template <typename T>
class DeviceBuffer
{
    public:
        DeviceBuffer() noexcept = default;


        explicit DeviceBuffer(std::size_t count)
        {
            allocate(count);
        }


        ~DeviceBuffer() noexcept
        {
            reset();
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : data_(std::exchange(other.data_, nullptr)),
            size_(std::exchange(other.size_, 0))
        {
        }


        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
        {
            if (this == &other)
                return *this;

            reset();

            data_ = std::exchange(other.data_,nullptr);
            size_ = std::exchange(other.size_, 0);
            return *this;
        }


        void allocate(std::size_t count)
        {
            if (data_ != nullptr)
                throw std::logic_error("DeviceBuffer is already allocated");

            if (count == 0)
                return;

            if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
                throw std::overflow_error("DeviceBuffer allocation size overflows");

            T* newData = nullptr;

            const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&newData), count * sizeof(T));

            if (error != cudaSuccess)
                throw std::runtime_error(std::string("DeviceBuffer cudaMalloc failed: ") + cudaGetErrorString(error));

            data_ = newData;
            size_ = count;
        }


        void reset() noexcept
        {
            if (data_ != nullptr)
                (void)cudaFree(data_);

            data_ = nullptr;
            size_ = 0;
        }


        [[nodiscard]]
        T* data() noexcept
        {
            return data_;
        }


        [[nodiscard]]
        const T* data() const noexcept
        {
            return data_;
        }


        [[nodiscard]]
        std::size_t size() const noexcept
        {
            return size_;
        }


        [[nodiscard]]
        bool empty() const noexcept
        {
            return size_ == 0;
        }


        void swap(DeviceBuffer& other) noexcept
        {
            std::swap(data_, other.data_);
            std::swap(size_, other.size_);
        }


    private:
        T* data_ = nullptr;
        std::size_t size_ = 0;
};