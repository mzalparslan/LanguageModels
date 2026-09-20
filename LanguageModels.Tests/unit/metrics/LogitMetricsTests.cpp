#include "pch.h"
#include "BasicGPT.h"
#include "LogitMetrics.h"
#include "TestSupport.h"
#include <cmath>
#include <limits>

using testsupport::makeMatrix;

// ------------------------------------------------------------------ scoreLogits

TEST(LogitMetricsTest, ScoresAHandComputedTwoRowExample) {
    // Both rows hold the logits 1, 2, 3. Row 0 targets token 2 (the arg-max),
    // row 1 targets token 0.
    auto logits = makeMatrix(2, 3, { 1, 2, 3, 1, 2, 3 });
    double logSumExp = std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0));

    Metrics metrics = evaluation::scoreLogits(logits, { 2, 0 });

    EXPECT_NEAR(metrics.loss, ((logSumExp - 3.0) + (logSumExp - 1.0)) / 2.0, 1e-12);
    EXPECT_NEAR(metrics.perplexity, std::exp(metrics.loss), 1e-12);
    EXPECT_DOUBLE_EQ(metrics.accuracy, 0.5);
    EXPECT_NEAR(metrics.bpc, metrics.loss / std::log(2.0), 1e-12);
}

TEST(LogitMetricsTest, EqualLogitsGiveAPerplexityEqualToTheVocabularySize) {
    auto logits = makeMatrix(3, 5, std::vector<double>(15, 0.7));

    Metrics metrics = evaluation::scoreLogits(logits, { 0, 3, 4 });

    EXPECT_NEAR(metrics.perplexity, 5.0, 1e-12);
    EXPECT_NEAR(metrics.loss, std::log(5.0), 1e-12);
}

TEST(LogitMetricsTest, ConfidentCorrectPredictionsGivePerplexityOne) {
    auto logits = makeMatrix(2, 3, { 50, 0, 0, 0, 0, 50 });

    Metrics metrics = evaluation::scoreLogits(logits, { 0, 2 });

    EXPECT_NEAR(metrics.perplexity, 1.0, 1e-9);
    EXPECT_DOUBLE_EQ(metrics.accuracy, 1.0);
}

TEST(LogitMetricsTest, HugeLogitsDoNotOverflow) {
    // exp(1000) overflows a double; the log-softmax must not compute it directly.
    auto logits = makeMatrix(1, 2, { 1000, 0 });

    Metrics metrics = evaluation::scoreLogits(logits, { 0 });

    EXPECT_TRUE(std::isfinite(metrics.loss));
    EXPECT_NEAR(metrics.loss, 0.0, 1e-12);
}

TEST(LogitMetricsTest, AllTiedLogitsCountOnlyTheLowestTokenAsCorrect) {
    auto logits = makeMatrix(2, 3, std::vector<double>(6, 1.0));

    Metrics metrics = evaluation::scoreLogits(logits, { 0, 1 });

    EXPECT_DOUBLE_EQ(metrics.accuracy, 0.5);
}

TEST(LogitMetricsTest, WorksForFloatLogits) {
    Tensor<float> logits({ 1, 2 });
    logits[0] = 0.0f;
    logits[1] = 0.0f;

    Metrics metrics = evaluation::scoreLogits(logits, { 1 });

    EXPECT_NEAR(metrics.loss, std::log(2.0), 1e-6);
}

TEST(LogitMetricsTest, ATargetWithZeroProbabilityThrowsNonFiniteError) {
    double inf = std::numeric_limits<double>::infinity();
    auto logits = makeMatrix(1, 2, { 0, -inf });

    EXPECT_THROW(evaluation::scoreLogits(logits, { 1 }), NonFiniteError);
}

TEST(LogitMetricsTest, RejectsInvalidInput) {
    auto logits = makeMatrix(2, 3, { 1, 2, 3, 1, 2, 3 });

    EXPECT_THROW(evaluation::scoreLogits(logits, { 0 }), InvalidSizeError);
    EXPECT_THROW(evaluation::scoreLogits(logits, { 0, 3 }), InvalidParameterError);
    EXPECT_THROW(evaluation::scoreLogits(Tensor<double>(), std::vector<std::size_t>{}), InvalidSizeError);
    EXPECT_THROW(evaluation::scoreLogits(Tensor<double>({ 4 }), { 0, 0, 0, 0 }), InvalidSizeError);
}

// ------------------------------------------------------------ MetricsAccumulator

TEST(MetricsAccumulatorTest, BatchesAreWeightedBySizeNotAveraged) {
    // One row and three rows scored separately must equal all four scored together.
    auto all = makeMatrix(4, 3, { 1, 2, 3, 3, 2, 1, 0, 5, 0, 1, 1, 4 });
    auto first = makeMatrix(1, 3, { 1, 2, 3 });
    auto rest = makeMatrix(3, 3, { 3, 2, 1, 0, 5, 0, 1, 1, 4 });
    std::vector<std::size_t> targets = { 2, 1, 1, 0 };

    evaluation::MetricsAccumulator accumulator;
    accumulator.addLogits(first, { 2 });
    accumulator.addLogits(rest, { 1, 1, 0 });

    Metrics together = evaluation::scoreLogits(all, targets);
    Metrics split = accumulator.result();

    EXPECT_EQ(accumulator.count(), 4u);
    EXPECT_NEAR(split.loss, together.loss, 1e-12);
    EXPECT_DOUBLE_EQ(split.accuracy, together.accuracy);
}

TEST(MetricsAccumulatorTest, AddsRawTotals) {
    evaluation::MetricsAccumulator accumulator;
    accumulator.add(10.0, 3, 5);
    accumulator.add(10.0, 4, 5);

    Metrics metrics = accumulator.result();

    EXPECT_DOUBLE_EQ(metrics.loss, 2.0);
    EXPECT_DOUBLE_EQ(metrics.accuracy, 0.7);
}

TEST(MetricsAccumulatorTest, ResultOfNothingThrows) {
    evaluation::MetricsAccumulator accumulator;

    EXPECT_THROW(accumulator.result(), DivisionByZeroError);
}

TEST(MetricsAccumulatorTest, MoreCorrectThanPredictionsThrows) {
    evaluation::MetricsAccumulator accumulator;

    EXPECT_THROW(accumulator.add(1.0, 6, 5), InvalidParameterError);
}

// ------------------------------------------------------------------ topKAccuracy

TEST(TopKAccuracyTest, OneIsOrdinaryAccuracy) {
    auto logits = makeMatrix(3, 3, { 3, 2, 1, 1, 3, 2, 1, 2, 3 });

    EXPECT_DOUBLE_EQ(evaluation::topKAccuracy(logits, { 0, 1, 0 }, 1), 2.0 / 3.0);
}

TEST(TopKAccuracyTest, LargerKAcceptsRunnerUps) {
    // Row 0 ranks token 1 second, row 1 ranks token 0 last.
    auto logits = makeMatrix(2, 3, { 3, 2, 1, 1, 3, 2 });

    EXPECT_DOUBLE_EQ(evaluation::topKAccuracy(logits, { 1, 0 }, 1), 0.0);
    EXPECT_DOUBLE_EQ(evaluation::topKAccuracy(logits, { 1, 0 }, 2), 0.5);
    EXPECT_DOUBLE_EQ(evaluation::topKAccuracy(logits, { 1, 0 }, 3), 1.0);
}

TEST(TopKAccuracyTest, TiesAreResolvedInTheModelsFavour) {
    auto logits = makeMatrix(1, 3, { 1, 1, 1 });

    EXPECT_DOUBLE_EQ(evaluation::topKAccuracy(logits, { 2 }, 1), 1.0);
}

TEST(TopKAccuracyTest, RejectsInvalidInput) {
    auto logits = makeMatrix(2, 3, { 1, 2, 3, 3, 2, 1 });

    EXPECT_THROW(evaluation::topKAccuracy(logits, { 0, 0 }, 0), InvalidParameterSizeError);
    EXPECT_THROW(evaluation::topKAccuracy(logits, { 0, 0 }, 4), InvalidSizeError);
    EXPECT_THROW(evaluation::topKAccuracy(logits, { 0 }, 1), InvalidSizeError);
    EXPECT_THROW(evaluation::topKAccuracy(logits, { 0, 3 }, 1), InvalidParameterError);
}

// ------------------------------------------------------ DecoderOnlyModel::evaluate

namespace {
    const std::size_t modelVocab = 10;
    const std::vector<std::size_t> pattern = { 0, 1, 2, 0, 1 };
    const std::vector<std::size_t> patternTargets = { 1, 2, 0, 1, 2 };
}

TEST(DecoderOnlyModelEvaluateTest, MatchesScoringItsOwnLogits) {
    BasicGPT gpt(modelVocab, BasicGPTConfig::d_head, 1, BasicGPTConfig::maxSeqLen);
    Tensor<double> logits;
    gpt.forward(pattern, logits);

    Metrics expected = evaluation::scoreLogits(logits, patternTargets);
    Metrics actual = gpt.evaluate(pattern, patternTargets);

    EXPECT_DOUBLE_EQ(actual.loss, expected.loss);
    EXPECT_DOUBLE_EQ(actual.accuracy, expected.accuracy);
}

TEST(DecoderOnlyModelEvaluateTest, DoesNotChangeTheModel) {
    BasicGPT gpt(modelVocab, BasicGPTConfig::d_head, 1, BasicGPTConfig::maxSeqLen);

    Metrics first = gpt.evaluate(pattern, patternTargets);
    Metrics second = gpt.evaluate(pattern, patternTargets);

    EXPECT_DOUBLE_EQ(first.loss, second.loss);
}

TEST(DecoderOnlyModelEvaluateTest, ReportsImprovementAfterTraining) {
    BasicGPT gpt(modelVocab, BasicGPTConfig::d_head, 1, BasicGPTConfig::maxSeqLen);
    Metrics before = gpt.evaluate(pattern, patternTargets);

    for (int step = 0; step < 100; step++) {
        gpt.trainStep(pattern, patternTargets, 0.05);
    }
    Metrics after = gpt.evaluate(pattern, patternTargets);

    EXPECT_LT(after.loss, before.loss);
    EXPECT_LT(after.perplexity, before.perplexity);
    EXPECT_GE(after.accuracy, before.accuracy);
}

TEST(DecoderOnlyModelEvaluateTest, RejectsInvalidInput) {
    BasicGPT gpt(modelVocab, BasicGPTConfig::d_head, 1, BasicGPTConfig::maxSeqLen);

    EXPECT_THROW(gpt.evaluate(pattern, { 1, 2 }), InvalidSizeError);
    EXPECT_THROW(gpt.evaluate({}, {}), InvalidSizeError);
    EXPECT_THROW(gpt.evaluate({ 0, 1 }, { 1, 99 }), InvalidParameterError);
}
