#pragma once

#include <string>
#include <string_view>

#include "BenchmarkTimer.h"
#include "Logger.h"

/**
 * @brief Measures a scope and logs its duration (at debug level) when the
 * scope ends, including when it ends by an exception.
 *
 * Logger must outlive timer.
 */
class ScopedBenchmarkTimer {
public:
    /**
     * @brief Starts timing a named scope.
     *
     * @param logger Logger used to report elapsed time.
     * @param scopeName Name of operation being measured.
     */
    explicit ScopedBenchmarkTimer(Logger& logger, std::string_view scopeName)
        : logger(logger), scopeName(scopeName) {
        timer.start();
    }

    ScopedBenchmarkTimer(const ScopedBenchmarkTimer&) = delete;
    ScopedBenchmarkTimer& operator=(const ScopedBenchmarkTimer&) = delete;
    ScopedBenchmarkTimer(ScopedBenchmarkTimer&&) = delete;
    ScopedBenchmarkTimer& operator=(ScopedBenchmarkTimer&&) = delete;

    /**
     * @brief Stops timer and logs elapsed duration.
     */
    ~ScopedBenchmarkTimer() noexcept {
        try {
            const double elapsedMilliseconds = static_cast<double>(timer.stop());
            logger.debug() << scopeName << " elapsed time: " << elapsedMilliseconds << " ms.";
        }
        catch (...) {
            // Exceptions must not escape from a destructor.
        }
    }

private:
    Logger& logger;
    std::string scopeName;
    BenchmarkTimer timer;
};
