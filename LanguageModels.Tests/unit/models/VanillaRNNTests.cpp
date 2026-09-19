#include "pch.h"
#include "VanillaRNN.h"
#include "TestSupport.h"
#include <cmath>
#include <cstdlib>
#include <limits>

namespace {
    const std::size_t hiddenSize = 8;
    const std::size_t vocabSize = 3;

    // "abcabc..." as ids 0 1 2 0 1 2 and the ids that follow each of them.
    const std::vector<int> inputs = { 0, 1, 2, 0, 1, 2 };
    const std::vector<int> targets = { 1, 2, 0, 1, 2, 0 };

    // Trains from the zero state on every step and returns first/last loss.
    std::pair<double, double> trainOnPattern(VanillaRNN<double>& rnn, int steps, double learningRate) {
        double first = 0.0, last = 0.0;
        for (int step = 0; step < steps; step++) {
            std::vector<double> state = rnn.getZeroState();
            last = rnn.trainStep(inputs, targets, state, learningRate);
            if (step == 0) {
                first = last;
            }
        }
        return { first, last };
    }
}

TEST(VanillaRNNTest, InvalidSizesThrowInvalidParameterSizeError) {
    EXPECT_THROW(VanillaRNN<double>(0, vocabSize), InvalidParameterSizeError);
    EXPECT_THROW(VanillaRNN<double>(hiddenSize, 0), InvalidParameterSizeError);
}

TEST(VanillaRNNTest, ZeroStateHasHiddenSizeZeros) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);

    auto state = rnn.getZeroState();

    ASSERT_EQ(state.size(), hiddenSize);
    for (double value : state) {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }
}

TEST(VanillaRNNTest, SameSeedGivesTheSameFirstLoss) {
    VanillaRNN<double> first(hiddenSize, vocabSize, 7);
    VanillaRNN<double> second(hiddenSize, vocabSize, 7);
    VanillaRNN<double> other(hiddenSize, vocabSize, 8);
    auto stateA = first.getZeroState(), stateB = second.getZeroState(), stateC = other.getZeroState();

    double lossA = first.trainStep(inputs, targets, stateA, 0.1);
    double lossB = second.trainStep(inputs, targets, stateB, 0.1);
    double lossC = other.trainStep(inputs, targets, stateC, 0.1);

    EXPECT_DOUBLE_EQ(lossA, lossB);
    EXPECT_NE(lossA, lossC);
}

TEST(VanillaRNNTest, UntrainedModelIsAboutAsUncertainAsAUniformGuess) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);

    auto metrics = rnn.evaluate(inputs, targets, rnn.getZeroState());

    // Initial weights are tiny, so predictions are nearly uniform: loss ~ ln(vocab).
    EXPECT_NEAR(metrics.loss, std::log(static_cast<double>(vocabSize)), 0.05);
    EXPECT_NEAR(metrics.perplexity, static_cast<double>(vocabSize), 0.2);
}

TEST(VanillaRNNTest, TrainStepAdvancesTheHiddenState) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);
    auto state = rnn.getZeroState();

    (void)rnn.trainStep(inputs, targets, state, 0.1);

    ASSERT_EQ(state.size(), hiddenSize);
    bool anyNonZero = false;
    for (double value : state) {
        anyNonZero = anyNonZero || value != 0.0;
        EXPECT_LE(std::fabs(value), 1.0); // tanh output
    }
    EXPECT_TRUE(anyNonZero);
}

TEST(VanillaRNNTest, TrainingOnARepeatingPatternReducesTheLoss) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);

    auto [first, last] = trainOnPattern(rnn, 300, 0.1);

    EXPECT_LT(last, 0.3 * first);
}

TEST(VanillaRNNTest, EvaluateReportsImprovementAfterTraining) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);
    auto before = rnn.evaluate(inputs, targets, rnn.getZeroState());

    trainOnPattern(rnn, 300, 0.1);
    auto after = rnn.evaluate(inputs, targets, rnn.getZeroState());

    EXPECT_LT(after.loss, before.loss);
    EXPECT_LT(after.perplexity, before.perplexity);
    EXPECT_GT(after.accuracy, 0.9);
    EXPECT_LT(after.bpc, before.bpc);
}

TEST(VanillaRNNTest, EvaluateMetricsAreInternallyConsistent) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);
    trainOnPattern(rnn, 50, 0.1);

    auto metrics = rnn.evaluate(inputs, targets, rnn.getZeroState());

    EXPECT_NEAR(metrics.perplexity, std::exp(metrics.loss), 1e-9);
    EXPECT_NEAR(metrics.bpc, metrics.loss / std::log(2.0), 1e-9);
    EXPECT_GE(metrics.accuracy, 0.0);
    EXPECT_LE(metrics.accuracy, 1.0);
}

TEST(VanillaRNNTest, EvaluateDoesNotChangeTheModel) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);

    auto first = rnn.evaluate(inputs, targets, rnn.getZeroState());
    auto second = rnn.evaluate(inputs, targets, rnn.getZeroState());

    EXPECT_DOUBLE_EQ(first.loss, second.loss);
}

TEST(VanillaRNNTest, SampleReturnsSeedFollowedByRequestedCharacters) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);

    auto sample = rnn.sample(rnn.getZeroState(), 1, 10);

    ASSERT_EQ(sample.size(), 11u);
    EXPECT_EQ(sample[0], 1u);
    for (std::size_t id : sample) {
        EXPECT_LT(id, vocabSize);
    }
}

TEST(VanillaRNNTest, SampleOfZeroCharactersReturnsOnlyTheSeed) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);

    auto sample = rnn.sample(rnn.getZeroState(), 2, 0);

    EXPECT_EQ(sample, (std::vector<std::size_t>{ 2 }));
}

TEST(VanillaRNNTest, SampleIsReproducibleWhenTheCRandomGeneratorIsReseeded) {
    // sample() draws from the C library's rand(), so reseeding makes it repeatable.
    VanillaRNN<double> rnn(hiddenSize, vocabSize);

    std::srand(123);
    auto first = rnn.sample(rnn.getZeroState(), 0, 20);
    std::srand(123);
    auto second = rnn.sample(rnn.getZeroState(), 0, 20);

    EXPECT_EQ(first, second);
}

TEST(VanillaRNNTest, TrainedModelSamplesTheLearnedCycleMostOfTheTime) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);
    trainOnPattern(rnn, 500, 0.1);

    std::srand(1);
    auto sample = rnn.sample(rnn.getZeroState(), 0, 30);

    std::size_t followsCycle = 0;
    for (std::size_t i = 1; i < sample.size(); i++) {
        if (sample[i] == (sample[i - 1] + 1) % vocabSize) {
            followsCycle++;
        }
    }
    EXPECT_GE(followsCycle, 25u);
}

TEST(VanillaRNNTest, TrainStepValidatesItsArguments) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);
    auto state = rnn.getZeroState();

    // Learning rate.
    EXPECT_THROW(rnn.trainStep(inputs, targets, state, 0.0), InvalidParameterError);
    EXPECT_THROW(rnn.trainStep(inputs, targets, state, std::nan("")), NaNError);
    // Sequence lengths.
    EXPECT_THROW(rnn.trainStep({}, {}, state, 0.1), InvalidSizeError);
    EXPECT_THROW(rnn.trainStep({ 0, 1 }, { 1 }, state, 0.1), InvalidSizeError);
    // Character ids.
    EXPECT_THROW(rnn.trainStep({ -1, 1 }, { 1, 2 }, state, 0.1), InvalidParameterError);
    EXPECT_THROW(rnn.trainStep({ 0, static_cast<int>(vocabSize) }, { 1, 2 }, state, 0.1), InvalidParameterError);
    EXPECT_THROW(rnn.trainStep({ 0, 1 }, { -1, 2 }, state, 0.1), InvalidParameterError);
    EXPECT_THROW(rnn.trainStep({ 0, 1 }, { 1, static_cast<int>(vocabSize) }, state, 0.1), InvalidParameterError);
}

TEST(VanillaRNNTest, TrainStepValidatesTheHiddenState) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);
    std::vector<double> wrongSize(hiddenSize + 1, 0.0);
    std::vector<double> hasNaN(hiddenSize, 0.0);
    hasNaN[3] = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(rnn.trainStep(inputs, targets, wrongSize, 0.1), InvalidSizeError);
    EXPECT_THROW(rnn.trainStep(inputs, targets, hasNaN, 0.1), NaNError);
}

TEST(VanillaRNNTest, EvaluateAndSampleValidateTheirArguments) {
    VanillaRNN<double> rnn(hiddenSize, vocabSize);
    auto state = rnn.getZeroState();
    std::vector<double> wrongSize(hiddenSize - 1, 0.0);

    EXPECT_THROW(rnn.evaluate({}, {}, state), InvalidSizeError);
    EXPECT_THROW(rnn.evaluate({ 0, 1 }, { 1 }, state), InvalidSizeError);
    EXPECT_THROW(rnn.evaluate({ 0, 1 }, { 1, 99 }, state), InvalidParameterError);
    EXPECT_THROW(rnn.evaluate(inputs, targets, wrongSize), InvalidSizeError);

    EXPECT_THROW(rnn.sample(wrongSize, 0, 5), InvalidSizeError);
    EXPECT_THROW(rnn.sample(state, vocabSize, 5), InvalidParameterError);
}
