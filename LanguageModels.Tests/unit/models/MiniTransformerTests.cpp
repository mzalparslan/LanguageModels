#include "pch.h"
#include "LogitMetrics.h"
#include "MiniTransformer.h"
#include "TestSupport.h"
#include <cmath>

namespace {
    const std::size_t sourceVocab = 12;
    const std::size_t targetVocab = 10;

    // Toy translation pair: teacher-forced decoder input and expected output
    // (the input shifted one position to left).
    const std::vector<std::size_t> source = { 3, 4, 5, 6 };
    const std::vector<std::size_t> decoderInput = { 1, 2, 3, 4, 5 };
    const std::vector<std::size_t> expectedOutput = { 2, 3, 4, 5, 0 };
}

TEST(MiniTransformerTest, ForwardGivesOneLogitRowPerDecoderToken) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    Tensor<double> logits;

    model.forward(source, decoderInput, logits);

    EXPECT_EQ(logits.shape, (std::vector<std::size_t>{ decoderInput.size(), targetVocab }));
    EXPECT_NO_THROW(validation::requireAllFinite(logits, "logits"));
}

TEST(MiniTransformerTest, InvalidVocabularySizesThrowInvalidParameterSizeError) {
    EXPECT_THROW(MiniTransformer<double>(0, targetVocab), InvalidParameterSizeError);
    EXPECT_THROW(MiniTransformer<double>(sourceVocab, 0), InvalidParameterSizeError);
}

TEST(MiniTransformerTest, SameSeedGivesIdenticalLogitsAndDifferentSeedsDiffer) {
    MiniTransformer<double> first(sourceVocab, targetVocab, 100, 5);
    MiniTransformer<double> second(sourceVocab, targetVocab, 100, 5);
    MiniTransformer<double> other(sourceVocab, targetVocab, 100, 6);
    Tensor<double> a, b, c;

    first.forward(source, decoderInput, a);
    second.forward(source, decoderInput, b);
    other.forward(source, decoderInput, c);

    EXPECT_TRUE(testsupport::tensorsEqual(a, b));
    EXPECT_FALSE(testsupport::tensorsEqual(a, c));
}

TEST(MiniTransformerTest, DecoderIsCausalOverTargetTokens) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    Tensor<double> before, after;
    std::vector<std::size_t> changed = decoderInput;
    changed.back() = 9;

    model.forward(source, decoderInput, before);
    model.forward(source, changed, after);

    for (std::size_t i = 0; i < (decoderInput.size() - 1) * targetVocab; i++) {
        EXPECT_NEAR(before[i], after[i], 1e-12) << "index " << i;
    }
}

TEST(MiniTransformerTest, EveryTargetPositionSeesTheWholeSourceSentence) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    Tensor<double> before, after;
    std::vector<std::size_t> changed = source;
    changed.back() = 11;

    model.forward(source, decoderInput, before);
    model.forward(changed, decoderInput, after);

    // Cross-attention: even first target position depends on last source token.
    bool firstRowChanged = false;
    for (std::size_t j = 0; j < targetVocab; j++) {
        firstRowChanged = firstRowChanged || std::fabs(before[j] - after[j]) > 1e-9;
    }
    EXPECT_TRUE(firstRowChanged);
}

TEST(MiniTransformerTest, ForwardValidatesItsInputs) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    Tensor<double> logits;

    EXPECT_THROW(model.forward({}, decoderInput, logits), InvalidSizeError);
    EXPECT_THROW(model.forward(source, {}, logits), InvalidSizeError);
    EXPECT_THROW(model.forward(std::vector<std::size_t>(MiniTransformerConfig::maxSeqLen + 1, 1),
        decoderInput, logits), InvalidSizeError);
    EXPECT_THROW(model.forward(source,
        std::vector<std::size_t>(MiniTransformerConfig::maxSeqLen + 1, 1), logits), InvalidSizeError);
    EXPECT_THROW(model.forward({ sourceVocab }, decoderInput, logits), InvalidParameterError);
    EXPECT_THROW(model.forward(source, { targetVocab }, logits), InvalidParameterError);
}

TEST(MiniTransformerTest, ForwardAcceptsExactlyMaxSeqLenTokens) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    Tensor<double> logits;
    std::vector<std::size_t> longSentence(MiniTransformerConfig::maxSeqLen, 1);

    EXPECT_NO_THROW(model.forward(longSentence, longSentence, logits));
}

TEST(MiniTransformerTest, TrainStepReturnsFiniteSummedLoss) {
    MiniTransformer<double> model(sourceVocab, targetVocab);

    double loss = model.trainStep(source, decoderInput, expectedOutput, 0.001, UpdateRule::adam(1));

    EXPECT_TRUE(std::isfinite(loss));
    // Summed over positions, so an untrained model is near positions * ln(vocab).
    EXPECT_GT(loss, 0.0);
}

TEST(MiniTransformerTest, TrainingOnOnePairReducesItsLoss) {
    MiniTransformer<double> model(sourceVocab, targetVocab);

    double first = 0.0, last = 0.0;
    for (std::size_t step = 1; step <= 60; step++) {
        last = model.trainStep(source, decoderInput, expectedOutput, 0.005, UpdateRule::adam(step));
        if (step == 1) {
            first = last;
        }
    }

    EXPECT_LT(last, 0.5 * first);
}

TEST(MiniTransformerTest, TrainedModelPredictsTheExpectedTokens) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    for (std::size_t step = 1; step <= 150; step++) {
        (void)model.trainStep(source, decoderInput, expectedOutput, 0.005, UpdateRule::adam(step));
    }
    Tensor<double> logits;

    model.forward(source, decoderInput, logits);

    for (std::size_t position = 0; position < expectedOutput.size(); position++) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < targetVocab; j++) {
            if (logits[position * targetVocab + j] > logits[position * targetVocab + best]) {
                best = j;
            }
        }
        EXPECT_EQ(best, expectedOutput[position]) << "position " << position;
    }
}

TEST(MiniTransformerTest, TrainStepValidatesItsArguments) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    auto rule = UpdateRule::adam(1);

    EXPECT_THROW(model.trainStep(source, decoderInput, expectedOutput, 0.0, rule), InvalidParameterError);
    EXPECT_THROW(model.trainStep(source, decoderInput, expectedOutput, -1.0, rule), InvalidParameterError);
    EXPECT_THROW(model.trainStep(source, decoderInput, expectedOutput, std::nan(""), rule), NaNError);
    // One label per decoder position.
    EXPECT_THROW(model.trainStep(source, decoderInput, { 1, 2 }, 0.001, rule), InvalidSizeError);
    // Labels must be target-vocabulary ids.
    EXPECT_THROW(model.trainStep(source, decoderInput, { 2, 3, 4, 5, targetVocab }, 0.001, rule),
        InvalidParameterError);
}

TEST(MiniTransformerTest, TrainStepWorksWithPlainSgdToo) {
    MiniTransformer<double> model(sourceVocab, targetVocab);

    EXPECT_NO_THROW(model.trainStep(source, decoderInput, expectedOutput, 0.01, UpdateRule::sgd()));
}

// === decoding
//
// toy pair above reads as: start token 1, then 2 3 4 5, then end token 0.

namespace {
    const std::size_t startToken = 1;
    const std::size_t endToken = 0;
    const std::vector<std::size_t> learnedTranslation = { 1, 2, 3, 4, 5, 0 };

    MiniTransformer<double> trainedModel() {
        MiniTransformer<double> model(sourceVocab, targetVocab);
        for (std::size_t step = 1; step <= 150; step++) {
            (void)model.trainStep(source, decoderInput, expectedOutput, 0.005, UpdateRule::adam(step));
        }
        return model;
    }
}

TEST(MiniTransformerDecodingTest, GreedyDecodingFollowsTheArgMaxOfForward) {
    MiniTransformer<double> model(sourceVocab, targetVocab);

    // Reference: same loop written out with forward().
    std::vector<std::size_t> expected = { startToken };
    for (int step = 0; step < 6; step++) {
        Tensor<double> logits;
        model.forward(source, expected, logits);
        const double* lastRow = &logits.data[(logits.shape[0] - 1) * targetVocab];
        std::size_t best = 0;
        for (std::size_t j = 1; j < targetVocab; j++) {
            if (lastRow[j] > lastRow[best]) {
                best = j;
            }
        }
        expected.push_back(best);
        if (best == endToken) {
            break;
        }
    }

    EXPECT_EQ(model.generate(source, startToken, endToken, 6), expected);
}

TEST(MiniTransformerDecodingTest, TrainedModelGeneratesItsTranslationAndStopsAtTheEndToken) {
    MiniTransformer<double> model = trainedModel();

    EXPECT_EQ(model.generate(source, startToken, endToken, 10), learnedTranslation);
}

TEST(MiniTransformerDecodingTest, GenerationStopsAfterMaxNewTokensWhenNoEndTokenComes) {
    MiniTransformer<double> model = trainedModel();

    // Cut short after three new tokens: 1 2 3 4, no end token yet.
    EXPECT_EQ(model.generate(source, startToken, endToken, 3), (std::vector<std::size_t>{ 1, 2, 3, 4 }));
}

TEST(MiniTransformerDecodingTest, ZeroNewTokensReturnsOnlyTheStartToken) {
    MiniTransformer<double> model(sourceVocab, targetVocab);

    EXPECT_EQ(model.generate(source, startToken, endToken, 0), (std::vector<std::size_t>{ startToken }));
    auto hypothesis = model.beamSearch(source, startToken, endToken, 0, 3);
    EXPECT_EQ(hypothesis.tokens, (std::vector<std::size_t>{ startToken }));
    EXPECT_FALSE(hypothesis.finished);
    EXPECT_DOUBLE_EQ(hypothesis.logProbability, 0.0);
}

TEST(MiniTransformerDecodingTest, DecodingDoesNotChangeTheModel) {
    MiniTransformer<double> model = trainedModel();

    auto first = model.generate(source, startToken, endToken, 10);
    auto second = model.generate(source, startToken, endToken, 10);

    EXPECT_EQ(first, second);
}

TEST(MiniTransformerDecodingTest, BeamSearchFindsTheLearnedTranslation) {
    MiniTransformer<double> model = trainedModel();

    auto hypothesis = model.beamSearch(source, startToken, endToken, 10, 3);

    EXPECT_EQ(hypothesis.tokens, learnedTranslation);
    EXPECT_TRUE(hypothesis.finished);
    // A trained model is confident: whole sentence is close to probability 1.
    EXPECT_LE(hypothesis.logProbability, 0.0);
    EXPECT_GT(hypothesis.logProbability, -1.0);
}

TEST(MiniTransformerDecodingTest, BeamOfWidthOneIsGreedyDecoding) {
    MiniTransformer<double> untrained(sourceVocab, targetVocab);
    MiniTransformer<double> trained = trainedModel();

    EXPECT_EQ(untrained.beamSearch(source, startToken, endToken, 6, 1).tokens,
        untrained.generate(source, startToken, endToken, 6));
    EXPECT_EQ(trained.beamSearch(source, startToken, endToken, 10, 1).tokens,
        trained.generate(source, startToken, endToken, 10));
}

TEST(MiniTransformerDecodingTest, BeamScoreIsTheSumOfTheTokenLogProbabilities) {
    MiniTransformer<double> model(sourceVocab, targetVocab);

    auto hypothesis = model.beamSearch(source, startToken, endToken, 4, 3);

    // Score returned tokens independently: decoder reads every token but
    // last and must predict every token but first.
    std::vector<std::size_t> input(hypothesis.tokens.begin(), hypothesis.tokens.end() - 1);
    std::vector<std::size_t> targets(hypothesis.tokens.begin() + 1, hypothesis.tokens.end());
    Tensor<double> logits;
    model.forward(source, input, logits);
    Metrics metrics = evaluation::scoreLogits(logits, targets);

    EXPECT_NEAR(hypothesis.logProbability, -metrics.loss * static_cast<double>(targets.size()), 1e-9);
}

TEST(MiniTransformerDecodingTest, RejectsInvalidArguments) {
    MiniTransformer<double> model(sourceVocab, targetVocab);

    EXPECT_THROW(model.generate({}, startToken, endToken, 3), InvalidSizeError);
    EXPECT_THROW(model.beamSearch({}, startToken, endToken, 3, 2), InvalidSizeError);
    // decoder sees at most MiniTransformerConfig::maxSeqLen tokens.
    EXPECT_THROW(model.generate(source, startToken, endToken, MiniTransformerConfig::maxSeqLen + 1), InvalidSizeError);
    EXPECT_THROW(model.beamSearch(source, startToken, endToken, MiniTransformerConfig::maxSeqLen + 1, 2), InvalidSizeError);
    EXPECT_THROW(model.beamSearch(source, startToken, endToken, 3, 0), InvalidParameterSizeError);
    // Token ids must be target-vocabulary ids.
    EXPECT_THROW(model.generate(source, targetVocab, endToken, 3), InvalidParameterError);
    EXPECT_THROW(model.generate(source, startToken, targetVocab, 3), InvalidParameterError);
    EXPECT_THROW(model.beamSearch(source, startToken, targetVocab, 3, 2), InvalidParameterError);
    EXPECT_THROW(model.generate({ sourceVocab }, startToken, endToken, 3), InvalidParameterError);
}
