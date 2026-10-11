#include "pch.h"
#include "MetalRuntime.h"
#include "MetalVocabularyHead.h"
#include "../gpu/HeadReference.h"
#include <cmath>
#include <iostream>
#include <limits>

// The Metal head is checked against the CPU code it replaces, computed in double
// precision. The head works in float (Metal has no double), so the two agree to about
// 1e-4 rather than to rounding, and the tolerances below say so. Tests that need a GPU
// report themselves skipped on a machine without one, which includes the virtual
// machines some CI services run on.
#define SKIP_WITHOUT_METAL() \
    if (!metal::isAvailable()) { \
        std::cout << "[  SKIPPED ] no Metal device, or this build is not for an Apple platform\n"; \
        return; \
    }

namespace {
    using namespace headreference;
    using Head = metal::VocabularyHead<float>;

    Head makeHead(const Weights& weights) {
        Head head(sourceVocab, targetVocab, width);
        head.uploadWeights(convert<float>(weights.source), convert<float>(weights.target),
            convert<float>(weights.projection), convert<float>(weights.bias));
        return head;
    }

    struct FloatWeights {
        std::vector<float> source, target, projection, bias;
    };

    FloatWeights downloadAll(const Head& head) {
        FloatWeights out;
        head.downloadWeights(out.source, out.target, out.projection, out.bias);
        return out;
    }

    void expectNear(const std::vector<float>& actual, const std::vector<double>& expected, double tolerance, const char* what) {
        ASSERT_EQ(actual.size(), expected.size()) << what;
        for (std::size_t i = 0; i < actual.size(); i++) {
            ASSERT_NEAR(actual[i], expected[i], tolerance) << what << ", index " << i;
        }
    }
}

// -------------------------------------------------------- weights and lookups

TEST(MetalVocabularyHeadTest, DownloadReturnsExactlyWhatWasUploaded) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);

    const FloatWeights back = downloadAll(head);

    EXPECT_EQ(back.source, convert<float>(weights.source));
    EXPECT_EQ(back.target, convert<float>(weights.target));
    EXPECT_EQ(back.projection, convert<float>(weights.projection));
    EXPECT_EQ(back.bias, convert<float>(weights.bias));
}

TEST(MetalVocabularyHeadTest, ReportsItsSizes) {
    SKIP_WITHOUT_METAL();
    Head head(sourceVocab, targetVocab, width);

    EXPECT_EQ(head.sourceVocab(), sourceVocab);
    EXPECT_EQ(head.targetVocab(), targetVocab);
    EXPECT_EQ(head.width(), width);
}

TEST(MetalVocabularyHeadTest, LookupsReturnTheRowsOfTheTables) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    const std::vector<std::size_t> sourceIds = { 3, 0, 49, 3 };
    const std::vector<std::size_t> targetIds = { 699, 5, 5, 0, 350 };

    std::vector<float> sourceRows, targetRows;
    head.lookupSource(sourceIds, sourceRows);
    head.lookupTarget(targetIds, targetRows);

    ASSERT_EQ(sourceRows.size(), sourceIds.size() * width);
    ASSERT_EQ(targetRows.size(), targetIds.size() * width);
    for (std::size_t r = 0; r < sourceIds.size(); r++) {
        for (std::size_t c = 0; c < width; c++) {
            EXPECT_EQ(sourceRows[r * width + c], static_cast<float>(weights.source[sourceIds[r] * width + c]));
        }
    }
    for (std::size_t r = 0; r < targetIds.size(); r++) {
        for (std::size_t c = 0; c < width; c++) {
            EXPECT_EQ(targetRows[r * width + c], static_cast<float>(weights.target[targetIds[r] * width + c]));
        }
    }
}

// --------------------------------------------------------------- output layer

TEST(MetalVocabularyHeadTest, OutputLayerMatchesTheCpuLossAndInputGradient) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    LinearLayer<double> layer = makeReferenceLayer(weights);
    const std::vector<double> decoderOutput = pattern(6 * width, 1.0, 0.7);
    const std::vector<std::size_t> labels = { 0, 699, 12, 350, 12, 5 };

    std::vector<double> expectedDx;
    const double expectedLoss = referenceOutputLayer(layer, decoderOutput, labels, expectedDx);
    head.zeroGradients();
    std::vector<float> dx;
    const float loss = head.outputLayer(convert<float>(decoderOutput), labels, dx);

    EXPECT_NEAR(loss, expectedLoss, 1e-4 * std::max(1.0, std::fabs(expectedLoss)));
    expectNear(dx, expectedDx, 1e-4, "decoder output gradient");
}

TEST(MetalVocabularyHeadTest, PlainGradientDescentMatchesTheCpuProjectionUpdate) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    LinearLayer<double> layer = makeReferenceLayer(weights);
    const std::vector<double> decoderOutput = pattern(5 * width, 1.0, 0.2);
    const std::vector<std::size_t> labels = { 1, 2, 3, 4, 5 };

    std::vector<double> ignored;
    referenceOutputLayer(layer, decoderOutput, labels, ignored);
    layer.update(0.05, UpdateRule::sgd());
    head.zeroGradients();
    std::vector<float> ignoredFloat;
    head.outputLayer(convert<float>(decoderOutput), labels, ignoredFloat);
    head.updateSgd(0.05f);

    const FloatWeights after = downloadAll(head);
    expectNear(after.projection, layer.W.value.data, 1e-5, "projection weights");
    expectNear(after.bias, layer.b.value.data, 1e-5, "projection bias");
    // The embedding tables had no gradient, so they did not move.
    EXPECT_EQ(after.source, convert<float>(weights.source));
    EXPECT_EQ(after.target, convert<float>(weights.target));
}

TEST(MetalVocabularyHeadTest, AdamOnTheProjectionMatchesTheCpuOverSeveralSteps) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    LinearLayer<double> layer = makeReferenceLayer(weights);
    const std::vector<double> decoderOutput = pattern(4 * width, 1.0, 0.9);
    const std::vector<std::size_t> labels = { 10, 20, 30, 40 };

    for (std::size_t step = 1; step <= 4; step++) {
        std::vector<double> ignored;
        layer.zeroGrad();
        referenceOutputLayer(layer, decoderOutput, labels, ignored);
        layer.update(0.01, UpdateRule::adam(step));

        std::vector<float> ignoredFloat;
        head.zeroGradients();
        head.outputLayer(convert<float>(decoderOutput), labels, ignoredFloat);
        head.updateAdam(0.01f, step);
    }

    const FloatWeights after = downloadAll(head);
    expectNear(after.projection, layer.W.value.data, 1e-4, "projection weights");
    expectNear(after.bias, layer.b.value.data, 1e-4, "projection bias");
}

TEST(MetalVocabularyHeadTest, TrainingOnTheGpuLowersTheLoss) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    const std::vector<float> decoderOutput = convert<float>(pattern(5 * width, 1.0, 0.5));
    const std::vector<std::size_t> labels = { 7, 8, 9, 10, 11 };

    float first = 0.0f, last = 0.0f;
    for (std::size_t step = 1; step <= 30; step++) {
        std::vector<float> dx;
        head.zeroGradients();
        last = head.outputLayer(decoderOutput, labels, dx);
        head.updateAdam(0.05f, step);
        if (step == 1) {
            first = last;
        }
    }

    EXPECT_LT(last, 0.2f * first);
}

// ------------------------------------------------------------ embedding tables

TEST(MetalVocabularyHeadTest, EmbeddingGradientsAreAddedRowByRowIncludingRepeatedWords) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    // Word 4 occurs three times, and the gradients are large enough to be clipped.
    const std::vector<std::size_t> ids = { 4, 9, 4, 0, 4 };
    const std::vector<double> rowGradients = pattern(ids.size() * width, 0.6, 1.1);

    head.zeroGradients();
    head.accumulateSourceGradient(ids, convert<float>(rowGradients));
    head.updateSgd(0.1f);

    const FloatWeights after = downloadAll(head);
    expectNear(after.source, referenceTableUpdate(weights.source, sourceVocab, ids, rowGradients, 1, false, 0.1),
        1e-6, "source table");
    // Only the touched rows changed.
    const std::vector<float> original = convert<float>(weights.source);
    for (std::size_t i = 0; i < original.size(); i++) {
        const std::size_t row = i / width;
        if (row != 4 && row != 9 && row != 0) {
            ASSERT_EQ(after.source[i], original[i]) << "index " << i;
        }
    }
}

TEST(MetalVocabularyHeadTest, AdamOnTheEmbeddingTablesMatchesTheCpuOverSeveralSteps) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    const std::vector<std::size_t> sourceIds = { 2, 7, 2, 30 };
    const std::vector<std::size_t> targetIds = { 100, 5, 699, 5, 0 };
    const std::vector<double> sourceGradients = pattern(sourceIds.size() * width, 2.0, 0.3);
    const std::vector<double> targetGradients = pattern(targetIds.size() * width, 0.02, 0.6);

    for (std::size_t step = 1; step <= 5; step++) {
        head.zeroGradients();
        head.accumulateSourceGradient(sourceIds, convert<float>(sourceGradients));
        head.accumulateTargetGradient(targetIds, convert<float>(targetGradients));
        head.updateAdam(0.01f, step);
    }

    const FloatWeights after = downloadAll(head);
    expectNear(after.source, referenceTableUpdate(weights.source, sourceVocab, sourceIds, sourceGradients, 5, true, 0.01),
        1e-4, "source table");
    expectNear(after.target, referenceTableUpdate(weights.target, targetVocab, targetIds, targetGradients, 5, true, 0.01),
        1e-4, "target table");
}

TEST(MetalVocabularyHeadTest, UploadingAgainStartsAdamFromScratch) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    const std::vector<std::size_t> ids = { 1, 2 };
    const std::vector<float> gradients = convert<float>(pattern(ids.size() * width, 0.5, 0.2));

    // Two steps, then start over from the same weights: the first step of the
    // second run must equal the first step of the first run (fresh moments).
    for (std::size_t step = 1; step <= 2; step++) {
        head.zeroGradients();
        head.accumulateSourceGradient(ids, gradients);
        head.updateAdam(0.01f, step);
    }
    head.uploadWeights(convert<float>(weights.source), convert<float>(weights.target),
        convert<float>(weights.projection), convert<float>(weights.bias));
    head.zeroGradients();
    head.accumulateSourceGradient(ids, gradients);
    head.updateAdam(0.01f, 1);

    const FloatWeights after = downloadAll(head);
    expectNear(after.source,
        referenceTableUpdate(weights.source, sourceVocab, ids, std::vector<double>(gradients.begin(), gradients.end()), 1, true, 0.01),
        1e-6, "source table");
}

// ---------------------------------------------------------------------- errors

TEST(MetalVocabularyHeadTest, RejectsInvalidSizesAndIds) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    std::vector<float> rows, dx;

    EXPECT_THROW(Head(0, 1, 1), InvalidParameterSizeError);
    EXPECT_THROW(Head(1, 0, 1), InvalidParameterSizeError);
    EXPECT_THROW(Head(1, 1, 0), InvalidParameterSizeError);

    EXPECT_THROW(head.uploadWeights({}, convert<float>(weights.target), convert<float>(weights.projection),
        convert<float>(weights.bias)), InvalidSizeError);
    EXPECT_THROW(head.uploadWeights(convert<float>(weights.source), convert<float>(weights.target),
        convert<float>(weights.projection), {}), InvalidSizeError);

    EXPECT_THROW(head.lookupSource({}, rows), InvalidSizeError);
    EXPECT_THROW(head.lookupSource({ sourceVocab }, rows), InvalidParameterError);
    EXPECT_THROW(head.lookupTarget({ targetVocab }, rows), InvalidParameterError);

    const std::vector<float> output = convert<float>(pattern(2 * width, 1.0, 0.0));
    EXPECT_THROW(head.outputLayer(output, {}, dx), InvalidSizeError);
    EXPECT_THROW(head.outputLayer(output, { 1, 2, 3 }, dx), InvalidSizeError);
    EXPECT_THROW(head.outputLayer(output, { 1, targetVocab }, dx), InvalidParameterError);

    EXPECT_THROW(head.accumulateSourceGradient({ 1, 2 }, std::vector<float>(3, 0.0f)), InvalidSizeError);
    EXPECT_THROW(head.accumulateTargetGradient({ targetVocab }, std::vector<float>(width, 0.0f)), InvalidParameterError);
}

TEST(MetalVocabularyHeadTest, RejectsInvalidOptimizerArguments) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);

    EXPECT_THROW(head.updateSgd(0.0f), InvalidParameterError);
    EXPECT_THROW(head.updateSgd(-1.0f), InvalidParameterError);
    EXPECT_THROW(head.updateAdam(0.0f, 1), InvalidParameterError);
    EXPECT_THROW(head.updateAdam(0.01f, 0), InvalidParameterError);
    EXPECT_THROW(head.updateAdam(std::nanf(""), 1), NaNError);
}

TEST(MetalVocabularyHeadTest, ALossThatIsNotFiniteIsReported) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    std::vector<float> output = convert<float>(pattern(2 * width, 1.0, 0.0));
    output[3] = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> dx;

    EXPECT_THROW(head.outputLayer(output, { 1, 2 }, dx), NaNError);
}

TEST(MetalVocabularyHeadTest, AGradientThatIsNotFiniteIsReportedByTheUpdate) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head head = makeHead(weights);
    std::vector<float> gradients = convert<float>(pattern(width, 0.5, 0.0));

    gradients[2] = std::numeric_limits<float>::quiet_NaN();
    head.zeroGradients();
    head.accumulateSourceGradient({ 3 }, gradients);
    EXPECT_THROW(head.updateAdam(0.01f, 1), NaNError);
    head.zeroGradients();
    head.accumulateSourceGradient({ 3 }, gradients);
    EXPECT_THROW(head.updateSgd(0.01f), NaNError);

    gradients[2] = std::numeric_limits<float>::infinity();
    head.zeroGradients();
    head.accumulateTargetGradient({ 3 }, gradients);
    EXPECT_THROW(head.updateAdam(0.01f, 1), NonFiniteError);

    // The NaN is now in the weights and in Adam's moments, exactly as it would be on the CPU,
    // so every later update reports it again until the weights are uploaded afresh.
    head.zeroGradients();
    EXPECT_THROW(head.updateAdam(0.01f, 2), NonFiniteError);
    head.uploadWeights(convert<float>(weights.source), convert<float>(weights.target),
        convert<float>(weights.projection), convert<float>(weights.bias));
    head.zeroGradients();
    EXPECT_NO_THROW(head.updateAdam(0.01f, 1));
}

// ------------------------------------------------------------------ ownership

TEST(MetalVocabularyHeadTest, AHeadCanBeMoved) {
    SKIP_WITHOUT_METAL();
    Weights weights;
    Head first = makeHead(weights);

    Head second = std::move(first);
    Head third(1, 1, 1);
    third = std::move(second);

    EXPECT_EQ(downloadAll(third).source, convert<float>(weights.source));
}

// ------------------------------------------------------------- where it cannot run

TEST(MetalVocabularyHeadTest, MetalCannotComputeInDouble) {
    // Metal has no 64-bit floating point, so a double head is refused on every platform.
    EXPECT_THROW(metal::VocabularyHead<double>(1, 1, 1), MetalError);
}

TEST(MetalVocabularyHeadTest, WithoutMetalAHeadCannotBeCreated) {
    if (metal::isAvailable()) {
        return; // only meaningful where there is nothing to use
    }

    EXPECT_THROW(Head(1, 1, 1), MetalError);
}
