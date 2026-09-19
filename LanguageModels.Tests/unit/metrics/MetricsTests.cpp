#include "pch.h"
#include "Metrics.h"
#include <cmath>
#include <limits>

TEST(MetricsTest, DerivesAllFourMetricsFromTotals) {
    // 10 predictions, summed loss 20 nats, 7 correct.
    Metrics metrics = Metrics::fromTotals(20.0, 7, 10);

    EXPECT_DOUBLE_EQ(metrics.loss, 2.0);
    EXPECT_NEAR(metrics.perplexity, std::exp(2.0), 1e-12);
    EXPECT_DOUBLE_EQ(metrics.accuracy, 0.7);
    EXPECT_NEAR(metrics.bpc, 2.0 / std::log(2.0), 1e-12);
}

TEST(MetricsTest, PerfectPredictionsGivePerplexityOneAndZeroBits) {
    Metrics metrics = Metrics::fromTotals(0.0, 5, 5);

    EXPECT_DOUBLE_EQ(metrics.loss, 0.0);
    EXPECT_DOUBLE_EQ(metrics.perplexity, 1.0);
    EXPECT_DOUBLE_EQ(metrics.accuracy, 1.0);
    EXPECT_DOUBLE_EQ(metrics.bpc, 0.0);
}

TEST(MetricsTest, UniformGuessOverVocabularyGivesPerplexityEqualToVocabularySize) {
    const double vocab = 26.0;
    // Cross-entropy of a uniform guess is ln(vocab) per prediction.
    Metrics metrics = Metrics::fromTotals(100 * std::log(vocab), 4, 100);

    EXPECT_NEAR(metrics.perplexity, vocab, 1e-9);
}

TEST(MetricsTest, BitsPerCharacterOfOneBitPerPrediction) {
    Metrics metrics = Metrics::fromTotals(std::log(2.0) * 8, 0, 8);

    EXPECT_NEAR(metrics.bpc, 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(metrics.accuracy, 0.0);
}

TEST(MetricsTest, ZeroPredictionsThrowsDivisionByZeroError) {
    EXPECT_THROW(Metrics::fromTotals(0.0, 0, 0), DivisionByZeroError);
}

TEST(MetricsTest, MoreCorrectThanTotalThrowsInvalidParameterError) {
    EXPECT_THROW(Metrics::fromTotals(1.0, 6, 5), InvalidParameterError);
}

TEST(MetricsTest, NaNLossThrowsNaNError) {
    EXPECT_THROW(Metrics::fromTotals(std::numeric_limits<double>::quiet_NaN(), 1, 2), NaNError);
}

TEST(MetricsTest, InfiniteLossThrowsNonFiniteError) {
    EXPECT_THROW(Metrics::fromTotals(std::numeric_limits<double>::infinity(), 1, 2), NonFiniteError);
}
