#include "pch.h"
#include "BenchmarkTimer.h"
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <type_traits>

TEST(BenchmarkTimerTest, StopReturnsUnsignedMilliseconds) {
    static_assert(std::is_same_v<decltype(std::declval<BenchmarkTimer&>().stop()), std::uint64_t>);
    SUCCEED();
}

TEST(BenchmarkTimerTest, MeasuresAtLeastTheSleptTime) {
    BenchmarkTimer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    std::uint64_t elapsed = timer.stop();

    EXPECT_GE(elapsed, 25u);
    // Generous upper bound so a loaded machine does not make this flaky.
    EXPECT_LT(elapsed, 5000u);
}

TEST(BenchmarkTimerTest, SubMillisecondIntervalsTruncateToZero) {
    BenchmarkTimer timer;

    timer.start();
    std::uint64_t elapsed = timer.stop();

    EXPECT_LT(elapsed, 5u);
}

TEST(BenchmarkTimerTest, DoubleStartThrowsLogicError) {
    BenchmarkTimer timer;
    timer.start();

    EXPECT_THROW(timer.start(), std::logic_error);
}

TEST(BenchmarkTimerTest, StopWithoutStartThrowsLogicError) {
    BenchmarkTimer timer;

    EXPECT_THROW((void)timer.stop(), std::logic_error);
}

TEST(BenchmarkTimerTest, StopTwiceThrowsLogicError) {
    BenchmarkTimer timer;
    timer.start();
    (void)timer.stop();

    EXPECT_THROW((void)timer.stop(), std::logic_error);
}

TEST(BenchmarkTimerTest, CanBeRestartedAfterStop) {
    BenchmarkTimer timer;

    timer.start();
    (void)timer.stop();

    EXPECT_NO_THROW(timer.start());
    EXPECT_NO_THROW((void)timer.stop());
}

TEST(BenchmarkTimerTest, IsNotCopyable) {
    static_assert(!std::is_copy_constructible_v<BenchmarkTimer>);
    static_assert(!std::is_copy_assignable_v<BenchmarkTimer>);
    SUCCEED();
}
