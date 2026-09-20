#include "pch.h"
#include "BasicGPT.h"
#include "BasicGPTWithMoE.h"
#include "TestSupport.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace {
    // BasicGPTConfig fixes the head width at 16 and the context at 20.
    const std::size_t vocabSize = 10;
    const std::size_t dModel = BasicGPTConfig::d_head;
    const std::size_t maxLen = BasicGPTConfig::maxSeqLen;

    // Concept probe: is DecoderOnlyModel<double, B, BasicGPTConfig> a valid type?
    template <typename B>
    concept CanBuildModel = requires { typename DecoderOnlyModel<double, B, BasicGPTConfig>; };

    // Trains on the repeating pattern 0, 1, 2, 0, 1, 2, ... and returns the
    // first and last loss.
    template <typename Model>
    std::pair<double, double> trainOnPattern(Model& model, int steps, double learningRate) {
        std::vector<std::size_t> input = { 0, 1, 2, 0, 1 };
        std::vector<std::size_t> target = { 1, 2, 0, 1, 2 };
        double first = 0.0, last = 0.0;
        for (int step = 0; step < steps; step++) {
            last = model.trainStep(input, target, learningRate);
            if (step == 0) {
                first = last;
            }
        }
        return { first, last };
    }
}

// -------------------------------------------------------------------- concept

TEST(DecoderOnlyModelTest, AcceptsOnlyTypesThatSatisfyTheDecoderBlockConcept) {
    static_assert(CanBuildModel<BasicDecoderBlock<double>>);
    static_assert(CanBuildModel<DecoderWithMoe<double>>);
    static_assert(!CanBuildModel<int>);
    static_assert(!CanBuildModel<LinearLayer<double>>);
    SUCCEED();
}

TEST(DecoderOnlyModelTest, NamedModelsAreAliasesOfTheGenericTemplate) {
    static_assert(std::is_same_v<BasicGPT,
        DecoderOnlyModel<double, BasicDecoderBlock<double>, BasicGPTConfig>>);
    static_assert(std::is_same_v<BasicGPTWithMoE,
        DecoderOnlyModel<double, DecoderWithMoe<double>, BasicGPTConfig>>);
    static_assert(std::is_same_v<GPTWithUnigram,
        DecoderOnlyModel<double, DecoderWithMoe<double>, GPTWithUnigramConfig>>);
    SUCCEED();
}

// === construction

TEST(DecoderOnlyModelTest, StoresItsConfiguration) {
    BasicGPT gpt(vocabSize, dModel, 3, maxLen);

    EXPECT_EQ(gpt.vocabSize, vocabSize);
    EXPECT_EQ(gpt.dModel, dModel);
    EXPECT_EQ(gpt.numLayers, 3u);
    EXPECT_EQ(gpt.maxLen, maxLen);
    EXPECT_EQ(gpt.layers.size(), 3u);
}

TEST(DecoderOnlyModelTest, InvalidSizesThrowInvalidParameterSizeError) {
    EXPECT_THROW(BasicGPT(0, dModel, 2, maxLen), InvalidParameterSizeError);
    EXPECT_THROW(BasicGPT(vocabSize, 0, 2, maxLen), InvalidParameterSizeError);
    EXPECT_THROW(BasicGPT(vocabSize, dModel, 0, maxLen), InvalidParameterSizeError);
    EXPECT_THROW(BasicGPT(vocabSize, dModel, 2, 0), InvalidParameterSizeError);
}

TEST(DecoderOnlyModelTest, ContextLongerThanConfigThrowsInvalidParameterSizeError) {
    EXPECT_NO_THROW(BasicGPT(vocabSize, dModel, 1, BasicGPTConfig::maxSeqLen));
    EXPECT_THROW(BasicGPT(vocabSize, dModel, 1, BasicGPTConfig::maxSeqLen + 1), InvalidParameterSizeError);
}

TEST(DecoderOnlyModelTest, ModelWidthMustMatchTheConfiguredHeadWidth) {
    // RoPE tables are sized from the config, so a different width fails on use.
    BasicGPT gpt(vocabSize, 8, 1, maxLen);
    Tensor<double> logits;

    EXPECT_THROW(gpt.forward({ 0, 1 }, logits), InvalidSizeError);
}

TEST(DecoderOnlyModelTest, MixtureModelRejectsMoreExpertsPerTokenThanExperts) {
    bool useMoe = true;
    std::size_t numExperts = 2, topK = 3;

    EXPECT_THROW(BasicGPTWithMoE(vocabSize, dModel, 1, maxLen, useMoe, numExperts, topK),
        InvalidParameterSizeError);
}

// --------------------------------------------------------------------- forward

TEST(DecoderOnlyModelTest, ForwardGivesOneLogitRowPerTokenAndOneColumnPerVocabularyEntry) {
    BasicGPT gpt(vocabSize, dModel, 2, maxLen);
    Tensor<double> logits;

    gpt.forward({ 0, 1, 2, 3 }, logits);

    EXPECT_EQ(logits.shape, (std::vector<std::size_t>{ 4, vocabSize }));
    EXPECT_NO_THROW(validation::requireAllFinite(logits, "logits"));
}

TEST(DecoderOnlyModelTest, ForwardIsDeterministicAcrossInstances) {
    // The default seed is fixed, so two models start with identical weights.
    BasicGPT first(vocabSize, dModel, 2, maxLen);
    BasicGPT second(vocabSize, dModel, 2, maxLen);
    Tensor<double> logitsA, logitsB;

    first.forward({ 3, 1, 4, 1, 5 }, logitsA);
    second.forward({ 3, 1, 4, 1, 5 }, logitsB);

    EXPECT_TRUE(testsupport::tensorsEqual(logitsA, logitsB));
}

TEST(DecoderOnlyModelTest, ForwardIsCausal) {
    BasicGPT gpt(vocabSize, dModel, 2, maxLen);
    Tensor<double> before, after;

    gpt.forward({ 3, 1, 4, 1, 5 }, before);
    gpt.forward({ 3, 1, 4, 1, 9 }, after);

    // Changing only the last token must not change earlier predictions.
    for (std::size_t i = 0; i < 4 * vocabSize; i++) {
        EXPECT_NEAR(before[i], after[i], 1e-12) << "index " << i;
    }
    bool lastRowChanged = false;
    for (std::size_t j = 0; j < vocabSize; j++) {
        lastRowChanged = lastRowChanged || std::fabs(before[4 * vocabSize + j] - after[4 * vocabSize + j]) > 1e-9;
    }
    EXPECT_TRUE(lastRowChanged);
}

TEST(DecoderOnlyModelTest, ForwardAcceptsAFullContextAndRejectsLonger) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);
    Tensor<double> logits;

    EXPECT_NO_THROW(gpt.forward(std::vector<std::size_t>(maxLen, 1), logits));
    EXPECT_THROW(gpt.forward(std::vector<std::size_t>(maxLen + 1, 1), logits), InvalidSizeError);
}

TEST(DecoderOnlyModelTest, ForwardRejectsEmptyInput) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);
    Tensor<double> logits;

    EXPECT_THROW(gpt.forward({}, logits), InvalidSizeError);
}

TEST(DecoderOnlyModelTest, ForwardRejectsTokenIdsOutsideTheVocabulary) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);
    Tensor<double> logits;

    EXPECT_THROW(gpt.forward({ 0, vocabSize }, logits), InvalidParameterError);
}

// ------------------------------------------------------------------- trainStep

TEST(DecoderOnlyModelTest, TrainStepReturnsAFinitePositiveLoss) {
    BasicGPT gpt(vocabSize, dModel, 2, maxLen);

    double loss = gpt.trainStep({ 0, 1, 2, 0, 1 }, { 1, 2, 0, 1, 2 }, 0.05);

    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GT(loss, 0.0);
    // Mean cross-entropy of an untrained model is on the order of ln(vocab).
    EXPECT_LT(loss, 3.0 * std::log(static_cast<double>(vocabSize)));
}

TEST(DecoderOnlyModelTest, TrainingOnARepeatingPatternReducesTheLoss) {
    BasicGPT gpt(vocabSize, dModel, 2, maxLen);

    auto [first, last] = trainOnPattern(gpt, 150, 0.05);

    EXPECT_LT(last, 0.5 * first);
}

TEST(DecoderOnlyModelTest, TrainedModelContinuesTheLearnedPattern) {
    BasicGPT gpt(vocabSize, dModel, 2, maxLen);
    trainOnPattern(gpt, 300, 0.05);

    std::vector<std::size_t> generated = gpt.generate({ 0, 1, 2, 0, 1 }, 4);

    // ... 0 1 2 0 1 -> 2 0 1 2
    ASSERT_EQ(generated.size(), 9u);
    EXPECT_EQ(generated[5], 2u);
    EXPECT_EQ(generated[6], 0u);
    EXPECT_EQ(generated[7], 1u);
    EXPECT_EQ(generated[8], 2u);
}

TEST(DecoderOnlyModelTest, TrainStepValidatesItsArguments) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);

    EXPECT_THROW(gpt.trainStep({ 0, 1 }, { 1, 2 }, 0.0), InvalidParameterError);
    EXPECT_THROW(gpt.trainStep({ 0, 1 }, { 1, 2 }, -0.1), InvalidParameterError);
    EXPECT_THROW(gpt.trainStep({ 0, 1 }, { 1, 2 }, std::nan("")), NaNError);
    // Targets must be as long as the input.
    EXPECT_THROW(gpt.trainStep({ 0, 1, 2 }, { 1, 2 }, 0.05), InvalidSizeError);
    // Token ids must be inside the vocabulary.
    EXPECT_THROW(gpt.trainStep({ 0, 1 }, { 1, vocabSize }, 0.05), InvalidParameterError);
    EXPECT_THROW(gpt.trainStep({ vocabSize, 1 }, { 1, 2 }, 0.05), InvalidParameterError);
    EXPECT_THROW(gpt.trainStep({}, {}, 0.05), InvalidSizeError);
}

TEST(DecoderOnlyModelTest, BasicModelReportsNoAuxiliaryLoss) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);

    gpt.trainStep({ 0, 1 }, { 1, 2 }, 0.05);

    EXPECT_DOUBLE_EQ(gpt.lastAuxLoss, 0.0);
}

// -------------------------------------------------------------------- generate

TEST(DecoderOnlyModelTest, GenerateAppendsTheRequestedNumberOfTokens) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);

    auto generated = gpt.generate({ 0, 1 }, 6);

    ASSERT_EQ(generated.size(), 8u);
    EXPECT_EQ(generated[0], 0u);
    EXPECT_EQ(generated[1], 1u);
    for (std::size_t token : generated) {
        EXPECT_LT(token, vocabSize);
    }
}

TEST(DecoderOnlyModelTest, GenerateZeroTokensReturnsThePrompt) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);

    auto generated = gpt.generate({ 4, 5, 6 }, 0);

    EXPECT_EQ(generated, (std::vector<std::size_t>{ 4, 5, 6 }));
}

TEST(DecoderOnlyModelTest, GenerationIsGreedyAndThereforeRepeatable) {
    BasicGPT gpt(vocabSize, dModel, 2, maxLen);

    auto first = gpt.generate({ 1, 2 }, 8);
    auto second = gpt.generate({ 1, 2 }, 8);

    EXPECT_EQ(first, second);
}

TEST(DecoderOnlyModelTest, GenerateMayFillTheWholeContext) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);

    // A 1-token prompt plus (maxLen) new tokens needs a context of exactly maxLen.
    EXPECT_NO_THROW(gpt.generate({ 1 }, maxLen));
    EXPECT_THROW(gpt.generate({ 1 }, maxLen + 1), InvalidSizeError);
}

TEST(DecoderOnlyModelTest, GenerateValidatesThePrompt) {
    BasicGPT gpt(vocabSize, dModel, 1, maxLen);

    EXPECT_THROW(gpt.generate({}, 3), InvalidSizeError);
    EXPECT_THROW(gpt.generate({ vocabSize }, 3), InvalidParameterError);
    EXPECT_THROW(gpt.generate(std::vector<std::size_t>(maxLen + 1, 1), 1), InvalidSizeError);
}

// ------------------------------------------------------- mixture-of-experts GPT

TEST(DecoderOnlyModelMoETest, ForwardAndTrainingWork) {
    bool useMoe = true;
    std::size_t numExperts = 4, topK = 2;
    BasicGPTWithMoE gpt(vocabSize, dModel, 2, maxLen, useMoe, numExperts, topK);
    Tensor<double> logits;

    gpt.forward({ 0, 1, 2 }, logits);
    auto [first, last] = trainOnPattern(gpt, 100, 0.05);

    EXPECT_EQ(logits.shape, (std::vector<std::size_t>{ 3, vocabSize }));
    EXPECT_TRUE(std::isfinite(last));
    EXPECT_LT(last, first);
}

TEST(DecoderOnlyModelMoETest, DenseModeMatchesTheBlockWithoutExperts) {
    bool useMoe = false;
    BasicGPTWithMoE gpt(vocabSize, dModel, 1, maxLen, useMoe);

    EXPECT_NE(gpt.layers[0].ff, nullptr);
    EXPECT_EQ(gpt.layers[0].moe, nullptr);
}

TEST(DecoderOnlyModelMoETest, MixtureModeIsCausal) {
    bool useMoe = true;
    std::size_t numExperts = 4, topK = 2;
    BasicGPTWithMoE gpt(vocabSize, dModel, 1, maxLen, useMoe, numExperts, topK);
    Tensor<double> before, after;

    gpt.forward({ 3, 1, 4, 1, 5 }, before);
    gpt.forward({ 3, 1, 4, 1, 9 }, after);

    for (std::size_t i = 0; i < 4 * vocabSize; i++) {
        EXPECT_NEAR(before[i], after[i], 1e-12) << "index " << i;
    }
}

TEST(DecoderOnlyModelMoETest, UnigramModelUsesItsOwnLargerConfig) {
    const std::size_t width = GPTWithUnigramConfig::d_head;
    bool useMoe = false;
    GPTWithUnigram gpt(vocabSize, width, 1, GPTWithUnigramConfig::maxSeqLen, useMoe);
    Tensor<double> logits;

    // 30 tokens fit in this config's context but not in BasicGPTConfig's.
    EXPECT_NO_THROW(gpt.forward(std::vector<std::size_t>(30, 1), logits));
    EXPECT_EQ(logits.shape, (std::vector<std::size_t>{ 30, vocabSize }));
}
