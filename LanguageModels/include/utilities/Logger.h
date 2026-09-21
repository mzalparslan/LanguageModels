#pragma once

#include <iostream>
#include <ostream>
#include <sstream>
#include <string_view>
#include <utility>

/**
 * @brief Severity levels supported by Logger.
 */
enum class LogLevel {
    Debug = 0,
    Info,
    Warning,
    Error,
    Critical
};

/**
 * @brief Simple stream-based logging: `logger.info() << "text " << value;`
 * writes one "[INFO] text value" line when statement ends.
 *
 * Same interface as Logger of MachineLearningModels project, written
 * header-only to fit this library.
 *
 * @warning Not safe to call from several threads at once. Pipelines log only
 * from thread that called them, never from inside a multi-threaded
 * training step.
 */
class Logger {
private:
    /**
     * @brief One log line under construction. Collects what is streamed into
     * it and writes it (if its level is enabled) when destroyed.
     */
    class LogEntry {
    public:
        LogEntry(Logger& logger, LogLevel level)
            : logger_(&logger), level_(level), enabled_(logger.isEnabled(level)) {}

        LogEntry(const LogEntry&) = delete;
        LogEntry& operator=(const LogEntry&) = delete;

        LogEntry(LogEntry&& other) noexcept
            : logger_(other.logger_), level_(other.level_), enabled_(other.enabled_),
            stream_(std::move(other.stream_)) {
            other.enabled_ = false;
        }
        LogEntry& operator=(LogEntry&& other) = delete;

        ~LogEntry() noexcept {
            if (false == enabled_) {
                return;
            }

            try {
                logger_->log(level_, stream_.str());
            }
            catch (...) {
                // Logging failures must not escape from a destructor.
            }
        }

        template <typename Value>
        LogEntry& operator<<(const Value& value) {
            if (true == enabled_) {
                stream_ << value;
            }

            return *this;
        }

    private:
        Logger* logger_;
        LogLevel level_;
        bool enabled_;
        std::ostringstream stream_;
    };

public:
    /**
     * @param minimumLevel Entries below this level are dropped.
     * @param output Stream entries are written to (std::clog by default).
     * It must outlive Logger.
     */
    explicit Logger(LogLevel minimumLevel = LogLevel::Info, std::ostream& output = std::clog)
        : minimumLevel(minimumLevel), output(&output) {}

    /**
     * @brief Shared logger used by callers that do not supply their own: one
     * program-wide instance at LogLevel::Info writing to std::clog.
     */
    [[nodiscard]]
    static Logger& instance() {
        static Logger shared;
        return shared;
    }

    [[nodiscard]] LogEntry debug() { return LogEntry(*this, LogLevel::Debug); }
    [[nodiscard]] LogEntry info() { return LogEntry(*this, LogLevel::Info); }
    [[nodiscard]] LogEntry warning() { return LogEntry(*this, LogLevel::Warning); }
    [[nodiscard]] LogEntry error() { return LogEntry(*this, LogLevel::Error); }
    [[nodiscard]] LogEntry critical() { return LogEntry(*this, LogLevel::Critical); }

    /**
     * @brief Writes one line if level is at or above minimum level.
     */
    void log(LogLevel level, std::string_view message) {
        if (false == isEnabled(level)) {
            return;
        }

        (*output) << '[' << levelName(level) << "] " << message << '\n';
    }

private:
    [[nodiscard]]
    bool isEnabled(LogLevel level) const noexcept {
        return level >= minimumLevel;
    }

    [[nodiscard]]
    static std::string_view levelName(LogLevel level) noexcept {
        switch (level) {
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warning:
            return "WARNING";
        case LogLevel::Error:
            return "ERROR";
        case LogLevel::Critical:
            return "CRITICAL";
        }

        return "UNKNOWN";
    }

    LogLevel minimumLevel;
    std::ostream* output;
};
