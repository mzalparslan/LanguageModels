#include "pch.h"
#include "MetalRuntime.h"
#include <iostream>
#include <string>

// The Metal runtime is only available on Apple platforms, and there only when the
// machine (or virtual machine) has a GPU to give. Everywhere else these functions are
// stand-ins that say so, and that is what is checked first.

TEST(MetalRuntimeTest, AvailabilityIsReportedWithoutThrowing) {
    EXPECT_NO_THROW((void)metal::isAvailable());
    EXPECT_NO_THROW((void)metal::describeDevice());
}

TEST(MetalRuntimeTest, OffApplePlatformsMetalIsNotAvailable) {
#if defined(__APPLE__)
    GTEST_SUCCEED() << "an Apple platform: availability depends on the machine";
#else
    EXPECT_FALSE(metal::isAvailable());
    EXPECT_THROW((void)metal::deviceInfo(), MetalError);
    EXPECT_EQ(metal::describeDevice(), "Metal is not available");
#endif
}

TEST(MetalRuntimeTest, WithoutAMetalDeviceTheInfoThrowsMetalError) {
    if (metal::isAvailable()) {
        return; // only meaningful where there is nothing to use
    }

    EXPECT_THROW((void)metal::deviceInfo(), MetalError);
    EXPECT_NE(metal::describeDevice().find("not available"), std::string::npos);
}

TEST(MetalRuntimeTest, ADeviceIsDescribedByNameAndMemory) {
    if (!metal::isAvailable()) {
        std::cout << "[  SKIPPED ] no Metal device\n";
        return;
    }

    const metal::DeviceInfo info = metal::deviceInfo();

    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.recommendedMemoryBytes, 0u);
    EXPECT_EQ(metal::describeDevice().rfind(info.name, 0), 0u) << metal::describeDevice();
    std::cout << "[   Metal  ] " << metal::describeDevice() << "\n";
}
