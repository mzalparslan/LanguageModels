#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>

/**
 * @brief Measures elapsed execution time for a code section.
 */
class BenchmarkTimer {
private:
	using Clock = std::chrono::steady_clock;
	using Milliseconds = std::chrono::milliseconds;

public:
	BenchmarkTimer() = default;
	// Prevent copy operations.
	BenchmarkTimer(const BenchmarkTimer&) = delete;
	BenchmarkTimer& operator=(const BenchmarkTimer&) = delete;

	/**
	 * @brief Starts timer.
	 *
	 * @throws std::logic_error If already running.
	 */
	void start() {
		if (true == isRunning) {
			throw std::logic_error(
				"BenchmarkTimer::start() called while already running!");
		}

		startTime = Clock::now();
		isRunning = true;
	}

	/**
	 * @brief Stops timer and returns elapsed duration as milliseconds.
	 * Fractional part is truncated, so anything under 1 ms reads as 0.
	 *
	 * @throws std::logic_error If timer has not been started.
	 */
	[[nodiscard]] std::uint64_t stop() {
		if (false == isRunning) {
			throw std::logic_error(
				"BenchmarkTimer::stop() called before start()!");
		}

		auto elapsed = std::chrono::duration_cast<Milliseconds>(Clock::now() - startTime);
		isRunning = false;

		// steady_clock never goes backwards, so count is never negative.
		return static_cast<std::uint64_t>(elapsed.count());
	}

private:
	// Start time of start() method call.
	Clock::time_point startTime;
	// Flag to check if start() called first.
	bool isRunning = false;
};
