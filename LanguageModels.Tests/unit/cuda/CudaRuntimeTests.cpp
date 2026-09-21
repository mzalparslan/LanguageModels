#include "pch.h"
#include "CudaRuntime.h"
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

// The CUDA tests need a GPU. On a machine without one (or in a build without the
// CUDA Toolkit) the GPU tests report themselves skipped and the rest still checks
// the fallback behaviour, so the suite passes everywhere.
#define SKIP_WITHOUT_GPU() \
    if (!cuda::isAvailable()) { \
        std::cout << "[  SKIPPED ] no CUDA device, or this build has no CUDA support\n"; \
        return; \
    }

namespace {
    // Values that are exact in float, so the GPU's fused multiply-add and the CPU's
    // separate multiply and add give identical results.
    std::vector<float> smallIntegers(std::size_t count, std::size_t modulus) {
        std::vector<float> values(count);
        for (std::size_t i = 0; i < count; i++) {
            values[i] = static_cast<float>(i % modulus);
        }
        return values;
    }
}

TEST(CudaRuntimeTest, ReportsAvailabilityWithoutThrowing) {
    EXPECT_NO_THROW((void)cuda::isAvailable());

    // Shown in the test output, so it is clear which GPU (if any) the tests ran on.
    std::cout << "[   CUDA   ] " << cuda::describeDevice() << "\n";
}

TEST(CudaRuntimeTest, DescribesTheDeviceWhenOneIsAvailable) {
    SKIP_WITHOUT_GPU();

    cuda::DeviceInfo info = cuda::deviceInfo();

    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.totalMemoryBytes, 0u);
    EXPECT_GE(info.computeMajor, 3);
    EXPECT_GT(info.multiprocessorCount, 0);
    EXPECT_NE(cuda::describeDevice().find(info.name), std::string::npos);
}

TEST(CudaRuntimeTest, WithoutADeviceTheQueriesFailCleanly) {
    if (cuda::isAvailable()) {
        return; // only meaningful where there is nothing to use
    }

    EXPECT_THROW((void)cuda::deviceInfo(), CudaError);
    EXPECT_EQ(cuda::describeDevice(), "CUDA is not available");
    std::vector<float> y = { 1.0f };
    EXPECT_THROW(cuda::saxpy(2.0f, { 1.0f }, y), CudaError);
}

TEST(CudaRuntimeTest, CudaErrorIsARuntimeError) {
    static_assert(std::is_base_of_v<std::runtime_error, CudaError>);
    SUCCEED();
}

// ------------------------------------------------------------------- saxpy

TEST(CudaSaxpyTest, ComputesYPlusAXOnALargeVector) {
    SKIP_WITHOUT_GPU();
    // 1,000,003 is not a multiple of the block size, so the last block is partly idle.
    const std::size_t count = 1000003;
    const std::vector<float> x = smallIntegers(count, 100);
    std::vector<float> y = smallIntegers(count, 7);
    const std::vector<float> original = y;

    cuda::saxpy(3.0f, x, y);

    for (std::size_t i = 0; i < count; i++) {
        ASSERT_EQ(y[i], 3.0f * x[i] + original[i]) << "index " << i;
    }
}

TEST(CudaSaxpyTest, HandlesSizesAroundTheBlockBoundary) {
    SKIP_WITHOUT_GPU();
    for (std::size_t count : { 1u, 2u, 255u, 256u, 257u, 511u, 512u, 513u }) {
        const std::vector<float> x = smallIntegers(count, 50);
        std::vector<float> y = smallIntegers(count, 9);
        const std::vector<float> original = y;

        cuda::saxpy(2.0f, x, y);

        for (std::size_t i = 0; i < count; i++) {
            ASSERT_EQ(y[i], 2.0f * x[i] + original[i]) << "count " << count << ", index " << i;
        }
    }
}

TEST(CudaSaxpyTest, AgreesWithTheCpuForArbitraryValuesToWithinOneRounding) {
    SKIP_WITHOUT_GPU();
    const std::size_t count = 5000;
    std::vector<float> x(count), y(count);
    for (std::size_t i = 0; i < count; i++) {
        x[i] = std::sin(0.01f * static_cast<float>(i));
        y[i] = std::cos(0.02f * static_cast<float>(i));
    }
    std::vector<float> expected(count);
    for (std::size_t i = 0; i < count; i++) {
        expected[i] = 0.7f * x[i] + y[i];
    }

    cuda::saxpy(0.7f, x, y);

    for (std::size_t i = 0; i < count; i++) {
        ASSERT_NEAR(y[i], expected[i], 1e-6f) << "index " << i;
    }
}

TEST(CudaSaxpyTest, ManyLaunchesInARowKeepWorking) {
    SKIP_WITHOUT_GPU();
    const std::vector<float> x(1000, 1.0f);
    std::vector<float> y(1000, 0.0f);

    // Allocates and frees device memory every time: no leak, no failure.
    for (int round = 0; round < 200; round++) {
        cuda::saxpy(1.0f, x, y);
    }

    EXPECT_EQ(y.front(), 200.0f);
    EXPECT_EQ(y.back(), 200.0f);
}

TEST(CudaSaxpyTest, AZeroScaleLeavesYUnchanged) {
    SKIP_WITHOUT_GPU();
    const std::vector<float> x = smallIntegers(300, 10);
    std::vector<float> y = smallIntegers(300, 13);
    const std::vector<float> original = y;

    cuda::saxpy(0.0f, x, y);

    EXPECT_EQ(y, original);
}

TEST(CudaSaxpyTest, EmptyVectorsAreANoOpInEveryBuild) {
    std::vector<float> y;

    EXPECT_NO_THROW(cuda::saxpy(1.0f, {}, y));
    EXPECT_TRUE(y.empty());
}

TEST(CudaSaxpyTest, VectorsOfDifferentLengthAreRejectedInEveryBuild) {
    std::vector<float> y = { 1.0f, 2.0f, 3.0f };

    EXPECT_THROW(cuda::saxpy(1.0f, { 1.0f, 2.0f }, y), InvalidSizeError);
}
