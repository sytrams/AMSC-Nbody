#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

class CudaTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int deviceCount = 0;
        const cudaError_t status =
            cudaGetDeviceCount(&deviceCount);

        if (status == cudaErrorNoDevice ||
            status == cudaErrorInsufficientDriver ||
            deviceCount == 0)
        {
            cudaGetLastError();
            GTEST_SKIP()
                << "No CUDA-capable GPU is available";
        }

        ASSERT_EQ(status, cudaSuccess)
            << "cudaGetDeviceCount failed: "
            << cudaGetErrorString(status);
    }
};
