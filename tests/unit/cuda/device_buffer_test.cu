#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <type_traits>
#include <utility>
#include <vector>

#include "cuda_test.hpp"
#include "device_buffer.hpp"


namespace
{

class DeviceBufferTest : public CudaTest
{
};


TEST_F(
    DeviceBufferTest,
    DefaultConstructionIsEmpty)
{
    DeviceBuffer<double> buffer;

    EXPECT_EQ(
        buffer.data(),
        nullptr);

    EXPECT_EQ(
        buffer.size(),
        0u);

    EXPECT_TRUE(
        buffer.empty());
}


TEST_F(
    DeviceBufferTest,
    AllocatesRequestedDeviceStorage)
{
    constexpr std::size_t count = 8;

    DeviceBuffer<double> buffer(count);

    ASSERT_NE(
        buffer.data(),
        nullptr);

    EXPECT_EQ(
        buffer.size(),
        count);

    EXPECT_FALSE(
        buffer.empty());
}


TEST_F(
    DeviceBufferTest,
    StoresAndRetrievesDeviceData)
{
    const std::vector<int> input{
        10,
        20,
        30,
        40};

    DeviceBuffer<int> buffer(
        input.size());

    ASSERT_EQ(
        cudaMemcpy(
            buffer.data(),
            input.data(),
            input.size() * sizeof(int),
            cudaMemcpyHostToDevice),
        cudaSuccess);

    std::vector<int> output(
        input.size(),
        0);

    ASSERT_EQ(
        cudaMemcpy(
            output.data(),
            buffer.data(),
            output.size() * sizeof(int),
            cudaMemcpyDeviceToHost),
        cudaSuccess);

    EXPECT_EQ(
        output,
        input);
}


TEST_F(
    DeviceBufferTest,
    MoveConstructionTransfersOwnership)
{
    DeviceBuffer<int> source(16);

    int* originalPointer =
        source.data();

    ASSERT_NE(
        originalPointer,
        nullptr);

    DeviceBuffer<int> destination(
        std::move(source));

    EXPECT_EQ(
        destination.data(),
        originalPointer);

    EXPECT_EQ(
        destination.size(),
        16u);

    EXPECT_EQ(
        source.data(),
        nullptr);

    EXPECT_EQ(
        source.size(),
        0u);

    EXPECT_TRUE(
        source.empty());
}


TEST_F(
    DeviceBufferTest,
    MoveAssignmentTransfersOwnership)
{
    DeviceBuffer<int> source(16);
    DeviceBuffer<int> destination(8);

    int* sourcePointer =
        source.data();

    ASSERT_NE(
        sourcePointer,
        nullptr);

    destination =
        std::move(source);

    EXPECT_EQ(
        destination.data(),
        sourcePointer);

    EXPECT_EQ(
        destination.size(),
        16u);

    EXPECT_EQ(
        source.data(),
        nullptr);

    EXPECT_EQ(
        source.size(),
        0u);
}


TEST_F(
    DeviceBufferTest,
    ResetReleasesOwnership)
{
    DeviceBuffer<double> buffer(32);

    ASSERT_NE(
        buffer.data(),
        nullptr);

    buffer.reset();

    EXPECT_EQ(
        buffer.data(),
        nullptr);

    EXPECT_EQ(
        buffer.size(),
        0u);

    EXPECT_TRUE(
        buffer.empty());

    /*
     * reset() must also be safe when called repeatedly.
     */
    buffer.reset();

    EXPECT_EQ(
        buffer.data(),
        nullptr);
}


TEST_F(
    DeviceBufferTest,
    CanAllocateAfterReset)
{
    DeviceBuffer<int> buffer(4);

    buffer.reset();

    buffer.allocate(12);

    EXPECT_NE(
        buffer.data(),
        nullptr);

    EXPECT_EQ(
        buffer.size(),
        12u);
}


TEST_F(
    DeviceBufferTest,
    RejectsSecondAllocationWithoutReset)
{
    DeviceBuffer<int> buffer(4);

    EXPECT_THROW(
        buffer.allocate(8),
        std::logic_error);
}


/*
 * These properties are central to the ownership model.
 *
 * A DeviceBuffer must never be copyable because that
 * would create two owners for one CUDA allocation.
 */
static_assert(
    !std::is_copy_constructible_v<
        DeviceBuffer<int>>);

static_assert(
    !std::is_copy_assignable_v<
        DeviceBuffer<int>>);

static_assert(
    std::is_move_constructible_v<
        DeviceBuffer<int>>);

static_assert(
    std::is_move_assignable_v<
        DeviceBuffer<int>>);


} // namespace