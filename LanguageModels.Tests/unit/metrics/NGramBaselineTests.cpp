#include "pch.h"
#include "NGramBaseline.h"
#include <cmath>

TEST(NGramBaselineTest, UnigramProbabilityIsTheRelativeFrequency) {
    NGramBaseline baseline(3, 1, 0.0);
    baseline.train({ 0, 0, 0, 1 });

    EXPECT_DOUBLE_EQ(baseline.probability(0, 0), 0.75);
    EXPECT_DOUBLE_EQ(baseline.probability(2, 1), 0.25);
    EXPECT_DOUBLE_EQ(baseline.probability(0, 2), 0.0);
}

TEST(NGramBaselineTest, BigramProbabilityIsConditionedOnThePreviousToken) {
    NGramBaseline baseline(3, 2, 0.0);
    baseline.train({ 0, 1, 0, 1, 0, 2 });

    EXPECT_DOUBLE_EQ(baseline.probability(0, 1), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(baseline.probability(0, 2), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(baseline.probability(1, 0), 1.0);
}

TEST(NGramBaselineTest, SmoothingGivesUnseenTokensAProbability) {
    NGramBaseline baseline(3, 1, 1.0);
    baseline.train({ 0, 1 });

    // (count + 1) / (tokens + smoothing * vocab) = 1 / 5 for token 2.
    EXPECT_DOUBLE_EQ(baseline.probability(0, 2), 1.0 / 5.0);
    EXPECT_DOUBLE_EQ(baseline.probability(0, 0), 2.0 / 5.0);
}

TEST(NGramBaselineTest, ProbabilitiesSumToOne) {
    NGramBaseline bigram(4, 2, 0.5);
    bigram.train({ 0, 1, 2, 3, 0, 1, 1, 2 });

    for (std::size_t previous = 0; previous < 4; previous++) {
        double sum = 0.0;
        for (std::size_t next = 0; next < 4; next++) {
            sum += bigram.probability(previous, next);
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST(NGramBaselineTest, TrainingAccumulatesAcrossCalls) {
    NGramBaseline once(2, 1, 0.0);
    once.train({ 0, 0, 1, 1 });
    NGramBaseline twice(2, 1, 0.0);
    twice.train({ 0, 0 });
    twice.train({ 1, 1 });

    EXPECT_DOUBLE_EQ(twice.probability(0, 0), once.probability(0, 0));
}

TEST(NGramBaselineTest, UnigramOverEqualFrequenciesHasPerplexityEqualToTheAlphabetSize) {
    NGramBaseline baseline(2, 1, 0.0);
    baseline.train({ 0, 0, 1, 1 });

    Metrics metrics = baseline.evaluate({ 0, 0, 1, 1 });

    EXPECT_NEAR(metrics.perplexity, 2.0, 1e-12);
    // Three predictions (0->0, 0->1, 1->1); "0" is predicted, so only the first is right.
    EXPECT_DOUBLE_EQ(metrics.accuracy, 1.0 / 3.0);
}

TEST(NGramBaselineTest, BigramMemorisesADeterministicSequence) {
    NGramBaseline baseline(2, 2, 0.0);
    baseline.train({ 0, 1, 0, 1, 0, 1 });

    Metrics metrics = baseline.evaluate({ 0, 1, 0, 1, 0, 1 });

    EXPECT_NEAR(metrics.loss, 0.0, 1e-12);
    EXPECT_NEAR(metrics.perplexity, 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(metrics.accuracy, 1.0);
}

TEST(NGramBaselineTest, BigramBeatsUnigramOnTextWithStructure) {
    std::vector<std::size_t> tokens;
    for (int i = 0; i < 50; i++) {
        tokens.push_back(0);
        tokens.push_back(1);
        tokens.push_back(2);
        tokens.push_back(1);
    }
    NGramBaseline unigram(3, 1, 1.0);
    NGramBaseline bigram(3, 2, 1.0);
    unigram.train(tokens);
    bigram.train(tokens);

    EXPECT_LT(bigram.evaluate(tokens).perplexity, unigram.evaluate(tokens).perplexity);
}

TEST(NGramBaselineTest, AnUnseenTargetWithoutSmoothingThrowsNonFiniteError) {
    NGramBaseline baseline(3, 1, 0.0);
    baseline.train({ 0, 1 });

    EXPECT_THROW(baseline.evaluate({ 0, 2 }), NonFiniteError);
}

TEST(NGramBaselineTest, RejectsInvalidInput) {
    EXPECT_THROW(NGramBaseline(0, 1), InvalidParameterSizeError);
    EXPECT_THROW(NGramBaseline(3, 0), InvalidParameterError);
    EXPECT_THROW(NGramBaseline(3, 3), InvalidParameterError);
    EXPECT_THROW(NGramBaseline(3, 1, -1.0), InvalidParameterError);

    NGramBaseline baseline(3, 1, 0.0);
    EXPECT_THROW(baseline.probability(0, 0), DivisionByZeroError);
    EXPECT_THROW(baseline.train({ 0, 3 }), InvalidParameterError);
    EXPECT_THROW(baseline.probability(3, 0), InvalidParameterError);
    EXPECT_THROW(baseline.probability(0, 3), InvalidParameterError);
    baseline.train({ 0, 1 });
    EXPECT_THROW(baseline.evaluate({ 0 }), InvalidSizeError);
    EXPECT_THROW(baseline.evaluate({ 0, 3 }), InvalidParameterError);
}
