#include "pch.h"
#include "BERT.h"
#include "TestSupport.h"
#include <cmath>

namespace {
    const std::size_t vocabSize = 20;
    const std::size_t dModel = 16;
    const std::size_t numLayers = 2;
    const std::size_t maxLen = 12;

    // One training example: 6 tokens, segment ids, masked positions have a
    // non-zero label (label 0 means "not masked").
    class Example {
    public:
        std::vector<std::size_t> tokens = { 2, 7, 4, 3, 9, 3 };
        std::vector<std::size_t> types = { 0, 0, 0, 0, 1, 1 };
        std::vector<std::size_t> mlmLabels = { 0, 0, 5, 0, 0, 8 };
        std::size_t nspLabel = 1;
    };
}

TEST(BertModelTest, StoresItsConfiguration) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);

    EXPECT_EQ(bert.vocabSize, vocabSize);
    EXPECT_EQ(bert.dModel, dModel);
    EXPECT_EQ(bert.maxLen, maxLen);
    EXPECT_EQ(bert.layers.size(), numLayers);
}

TEST(BertModelTest, InvalidSizesThrowInvalidParameterSizeError) {
    EXPECT_THROW(BertModel<double>(0, dModel, numLayers, maxLen), InvalidParameterSizeError);
    EXPECT_THROW(BertModel<double>(vocabSize, 0, numLayers, maxLen), InvalidParameterSizeError);
    EXPECT_THROW(BertModel<double>(vocabSize, dModel, 0, maxLen), InvalidParameterSizeError);
    EXPECT_THROW(BertModel<double>(vocabSize, dModel, numLayers, 0), InvalidParameterSizeError);
}

TEST(BertModelTest, EncoderOutputHasOneVectorPerToken) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;
    Tensor<double> encoded;

    bert.forwardEncoder(example.tokens, example.types, encoded);

    EXPECT_EQ(encoded.shape, (std::vector<std::size_t>{ example.tokens.size(), dModel }));
    EXPECT_NO_THROW(validation::requireAllFinite(encoded, "encoder output"));
}

TEST(BertModelTest, EncoderIsBidirectional) {
    // Unlike a decoder, an encoder lets early tokens see later ones.
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;
    Tensor<double> before, after;
    bert.forwardEncoder(example.tokens, example.types, before);

    example.tokens.back() = 15;
    bert.forwardEncoder(example.tokens, example.types, after);

    bool firstTokenChanged = false;
    for (std::size_t j = 0; j < dModel; j++) {
        firstTokenChanged = firstTokenChanged || std::fabs(before[j] - after[j]) > 1e-9;
    }
    EXPECT_TRUE(firstTokenChanged);
}

TEST(BertModelTest, SegmentIdsChangeTheRepresentation) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;
    Tensor<double> segmentA, segmentB;

    bert.forwardEncoder(example.tokens, std::vector<std::size_t>(6, 0), segmentA);
    bert.forwardEncoder(example.tokens, std::vector<std::size_t>(6, 1), segmentB);

    EXPECT_FALSE(testsupport::tensorsNear(segmentA, segmentB, 1e-9));
}

TEST(BertModelTest, SameSeedGivesIdenticalEncodingsAndDifferentSeedsDiffer) {
    BertModel<double> first(vocabSize, dModel, numLayers, maxLen, 7);
    BertModel<double> second(vocabSize, dModel, numLayers, maxLen, 7);
    BertModel<double> other(vocabSize, dModel, numLayers, maxLen, 8);
    Example example;
    Tensor<double> outA, outB, outC;

    first.forwardEncoder(example.tokens, example.types, outA);
    second.forwardEncoder(example.tokens, example.types, outB);
    other.forwardEncoder(example.tokens, example.types, outC);

    EXPECT_TRUE(testsupport::tensorsEqual(outA, outB));
    EXPECT_FALSE(testsupport::tensorsEqual(outA, outC));
}

TEST(BertModelTest, EncoderValidatesItsInputs) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Tensor<double> out;

    // Empty input.
    EXPECT_THROW(bert.forwardEncoder({}, {}, out), InvalidSizeError);
    // Segment ids must match the token count.
    EXPECT_THROW(bert.forwardEncoder({ 1, 2, 3 }, { 0, 0 }, out), InvalidSizeError);
    // Longer than the position table.
    EXPECT_THROW(bert.forwardEncoder(std::vector<std::size_t>(maxLen + 1, 1),
        std::vector<std::size_t>(maxLen + 1, 0), out), InvalidSizeError);
    // Token id outside the vocabulary.
    EXPECT_THROW(bert.forwardEncoder({ 1, vocabSize }, { 0, 0 }, out), InvalidParameterError);
    // There are only two segments.
    EXPECT_THROW(bert.forwardEncoder({ 1, 2 }, { 0, 2 }, out), InvalidParameterError);
}

TEST(BertModelTest, EncoderAcceptsExactlyMaxLenTokens) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Tensor<double> out;

    EXPECT_NO_THROW(bert.forwardEncoder(std::vector<std::size_t>(maxLen, 1),
        std::vector<std::size_t>(maxLen, 0), out));
}

TEST(BertModelTest, MaskedLanguageModelLogitsCoverTheWholeVocabulary) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;
    Tensor<double> logits;

    bert.predictMaskedLogits(example.tokens, example.types, logits);

    EXPECT_EQ(logits.shape, (std::vector<std::size_t>{ example.tokens.size(), vocabSize }));
    EXPECT_NO_THROW(validation::requireAllFinite(logits, "MLM logits"));
}

TEST(BertModelTest, TrainStepReturnsFinitePositiveLoss) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;

    double loss = bert.trainStep(example.tokens, example.types, example.mlmLabels,
        example.nspLabel, 0.001, UpdateRule::adam(1));

    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GT(loss, 0.0);
}

TEST(BertModelTest, TrainingOnOneExampleReducesItsLoss) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;

    double first = 0.0, last = 0.0;
    for (std::size_t step = 1; step <= 60; step++) {
        last = bert.trainStep(example.tokens, example.types, example.mlmLabels,
            example.nspLabel, 0.005, UpdateRule::adam(step));
        if (step == 1) {
            first = last;
        }
    }

    EXPECT_LT(last, 0.5 * first);
}

TEST(BertModelTest, TrainingLearnsTheNextSentenceLabel) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;
    example.mlmLabels.assign(example.tokens.size(), 0); // no masked tokens: NSP only

    double first = 0.0, last = 0.0;
    for (std::size_t step = 1; step <= 40; step++) {
        last = bert.trainStep(example.tokens, example.types, example.mlmLabels,
            example.nspLabel, 0.005, UpdateRule::adam(step));
        if (step == 1) {
            first = last;
        }
    }

    // With nothing masked the loss is the 2-class NSP loss alone; fitting
    // the label drives it well below where it started (and below ln 2).
    EXPECT_GT(first, 0.0);
    EXPECT_LT(last, 0.5 * first);
    EXPECT_LT(last, std::log(2.0));
}

TEST(BertModelTest, TrainStepWorksWithPlainSgdToo) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;

    EXPECT_NO_THROW(bert.trainStep(example.tokens, example.types, example.mlmLabels,
        example.nspLabel, 0.01, UpdateRule::sgd()));
}

TEST(BertModelTest, TrainStepValidatesItsArguments) {
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);
    Example example;
    auto rule = UpdateRule::adam(1);

    // Learning rate.
    EXPECT_THROW(bert.trainStep(example.tokens, example.types, example.mlmLabels, 1, 0.0, rule), InvalidParameterError);
    EXPECT_THROW(bert.trainStep(example.tokens, example.types, example.mlmLabels, 1, std::nan(""), rule), NaNError);
    // NSP label is a class index: 0 or 1.
    EXPECT_THROW(bert.trainStep(example.tokens, example.types, example.mlmLabels, 2, 0.001, rule), InvalidParameterError);
    // One MLM label per token.
    EXPECT_THROW(bert.trainStep(example.tokens, example.types, { 0, 0, 5 }, 1, 0.001, rule), InvalidSizeError);
    // MLM labels must be vocabulary ids.
    EXPECT_THROW(bert.trainStep(example.tokens, example.types, { 0, 0, vocabSize, 0, 0, 0 }, 1, 0.001, rule), InvalidParameterError);
}
