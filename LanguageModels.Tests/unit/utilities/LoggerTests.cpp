#include "pch.h"
#include "Logger.h"
#include "ScopedBenchmarkTimer.h"
#include <sstream>
#include <stdexcept>

TEST(LoggerTest, WritesOneLineWithTheLevelName) {
    std::ostringstream output;
    Logger logger(LogLevel::Debug, output);

    logger.debug() << "d";
    logger.info() << "i";
    logger.warning() << "w";
    logger.error() << "e";
    logger.critical() << "c";

    EXPECT_EQ(output.str(), "[DEBUG] d\n[INFO] i\n[WARNING] w\n[ERROR] e\n[CRITICAL] c\n");
}

TEST(LoggerTest, JoinsEverythingStreamedIntoOneEntryIntoOneLine) {
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);

    logger.info() << "epoch " << 3 << " loss " << 0.5;

    EXPECT_EQ(output.str(), "[INFO] epoch 3 loss 0.5\n");
}

TEST(LoggerTest, DropsEntriesBelowTheMinimumLevel) {
    std::ostringstream output;
    Logger logger(LogLevel::Warning, output);

    logger.debug() << "hidden";
    logger.info() << "hidden";
    logger.warning() << "shown";
    logger.error() << "shown too";

    EXPECT_EQ(output.str(), "[WARNING] shown\n[ERROR] shown too\n");
}

TEST(LoggerTest, TheDefaultLevelIsInfo) {
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);

    logger.debug() << "hidden";
    logger.info() << "shown";

    EXPECT_EQ(output.str(), "[INFO] shown\n");
}

TEST(LoggerTest, LogWritesAMessageDirectly) {
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);

    logger.log(LogLevel::Error, "direct");
    logger.log(LogLevel::Debug, "hidden");

    EXPECT_EQ(output.str(), "[ERROR] direct\n");
}

TEST(LoggerTest, TheSharedInstanceIsOneObject) {
    EXPECT_EQ(&Logger::instance(), &Logger::instance());
}

TEST(LoggerTest, AnEntryThatIsNeverStreamedToStillWritesAnEmptyLine) {
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);

    { auto entry = logger.info(); }

    EXPECT_EQ(output.str(), "[INFO] \n");
}

// ---------------------------------------------------- ScopedBenchmarkTimer

TEST(ScopedBenchmarkTimerTest, LogsTheScopeNameWhenTheScopeEnds) {
    std::ostringstream output;
    Logger logger(LogLevel::Debug, output);

    {
        ScopedBenchmarkTimer timer(logger, "loading");
        EXPECT_TRUE(output.str().empty()) << "nothing is logged while scope is running";
    }

    EXPECT_NE(output.str().find("[DEBUG] loading elapsed time: "), std::string::npos) << output.str();
    EXPECT_NE(output.str().find(" ms."), std::string::npos);
}

TEST(ScopedBenchmarkTimerTest, LogsEvenWhenTheScopeEndsWithAnException) {
    std::ostringstream output;
    Logger logger(LogLevel::Debug, output);

    try {
        ScopedBenchmarkTimer timer(logger, "failing step");
        throw std::runtime_error("boom");
    }
    catch (const std::runtime_error&) {
    }

    EXPECT_NE(output.str().find("failing step elapsed time"), std::string::npos);
}

TEST(ScopedBenchmarkTimerTest, IsSilentAtTheDefaultInfoLevel) {
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);

    { ScopedBenchmarkTimer timer(logger, "quiet"); }

    EXPECT_TRUE(output.str().empty());
}
