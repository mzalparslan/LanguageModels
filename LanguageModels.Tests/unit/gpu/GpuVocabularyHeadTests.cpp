#include "pch.h"
#include "GpuVocabularyHead.h"
#include "HeadReference.h"
#include <iostream>
#include <string>

// gpu::VocabularyHead picks the CUDA or the Metal head that is present and forwards to
// it. What the heads compute is tested in their own files (CudaVocabularyHeadTests.cpp,
// MetalVocabularyHeadTests.cpp); these tests check the choice of backend and that the
// wrapper forwards faithfully. Tests that need a GPU report themselves skipped on a
// machine without one.
#define SKIP_WITHOUT_GPU() \
    if (!gpu::isAvailable<float>()) { \
        std::cout << "[  SKIPPED ] no GPU (CUDA or Metal), or this build has none\n"; \
        return; \
    }

namespace {
    using namespace headreference;

    gpu::VocabularyHead<float> makeHead(const Weights& weights) {
        gpu::VocabularyHead<float> head(sourceVocab, targetVocab, width);
        head.uploadWeights(convert<float>(weights.source), convert<float>(weights.target),
            convert<float>(weights.projection), convert<float>(weights.bias));
        return head;
    }
}

// ------------------------------------------------------------- choosing a backend

TEST(GpuBackendTest, TheBackendsHaveNames) {
    EXPECT_STREQ(gpu::toString(gpu::Backend::None), "None");
    EXPECT_STREQ(gpu::toString(gpu::Backend::Cuda), "CUDA");
    EXPECT_STREQ(gpu::toString(gpu::Backend::Metal), "Metal");
}

TEST(GpuBackendTest, CudaIsPreferredAndMetalNeedsFloat) {
    const bool cudaPresent = cuda::isAvailable();
    const bool metalPresent = metal::isAvailable();

    if (cudaPresent) {
        EXPECT_EQ(gpu::availableBackend<float>(), gpu::Backend::Cuda);
        EXPECT_EQ(gpu::availableBackend<double>(), gpu::Backend::Cuda);
    }
    else {
        EXPECT_EQ(gpu::availableBackend<float>(), metalPresent ? gpu::Backend::Metal : gpu::Backend::None);
        // Metal has no 64-bit floating point.
        EXPECT_EQ(gpu::availableBackend<double>(), gpu::Backend::None);
    }
}

TEST(GpuBackendTest, IsAvailableAgreesWithTheBackend) {
    EXPECT_EQ(gpu::isAvailable<float>(), gpu::availableBackend<float>() != gpu::Backend::None);
    EXPECT_EQ(gpu::isAvailable<double>(), gpu::availableBackend<double>() != gpu::Backend::None);
}

TEST(GpuBackendTest, TheDeviceDescriptionNamesTheBackendOrSaysWhyThereIsNone) {
    const std::string floatDescription = gpu::describeDevice<float>();

    switch (gpu::availableBackend<float>()) {
    case gpu::Backend::Cuda:
        EXPECT_EQ(floatDescription.rfind("CUDA: ", 0), 0u) << floatDescription;
        break;
    case gpu::Backend::Metal:
        EXPECT_EQ(floatDescription.rfind("Metal: ", 0), 0u) << floatDescription;
        break;
    case gpu::Backend::None:
        EXPECT_NE(floatDescription.find("no GPU"), std::string::npos) << floatDescription;
        break;
    }
    EXPECT_FALSE(gpu::describeDevice<double>().empty());
}

TEST(GpuBackendTest, MetalAloneDoesNotServeADoubleModel) {
    if (cuda::isAvailable() || !metal::isAvailable()) {
        return; // only meaningful on a machine with Metal and no CUDA
    }

    EXPECT_THROW(gpu::VocabularyHead<double>(1, 1, 1), DeviceError);
    EXPECT_NE(gpu::describeDevice<double>().find("float"), std::string::npos);
}

// ------------------------------------------------------------- without a GPU

TEST(GpuVocabularyHeadTest, WithoutAGpuAHeadCannotBeCreated) {
    if (gpu::isAvailable<float>()) {
        return; // only meaningful where there is nothing to use
    }

    EXPECT_THROW(gpu::VocabularyHead<float>(1, 1, 1), DeviceError);
    EXPECT_THROW(gpu::VocabularyHead<double>(1, 1, 1), DeviceError);
}

// --------------------------------------------------------------- with a GPU

TEST(GpuVocabularyHeadTest, ReportsItsBackendAndSizes) {
    SKIP_WITHOUT_GPU();
    gpu::VocabularyHead<float> head(sourceVocab, targetVocab, width);

    EXPECT_EQ(head.backend(), gpu::availableBackend<float>());
    EXPECT_EQ(head.sourceVocab(), sourceVocab);
    EXPECT_EQ(head.targetVocab(), targetVocab);
    EXPECT_EQ(head.width(), width);
}

TEST(GpuVocabularyHeadTest, WeightsComeBackAsTheyWentIn) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    gpu::VocabularyHead<float> head = makeHead(weights);

    std::vector<float> source, target, projection, bias;
    head.downloadWeights(source, target, projection, bias);

    EXPECT_EQ(source, convert<float>(weights.source));
    EXPECT_EQ(target, convert<float>(weights.target));
    EXPECT_EQ(projection, convert<float>(weights.projection));
    EXPECT_EQ(bias, convert<float>(weights.bias));
}

TEST(GpuVocabularyHeadTest, LookupsReturnTheRowsOfTheTables) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    gpu::VocabularyHead<float> head = makeHead(weights);
    const std::vector<std::size_t> sourceIds = { 3, 0, 49, 3 };
    const std::vector<std::size_t> targetIds = { 699, 5, 0 };

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

TEST(GpuVocabularyHeadTest, OutputLayerMatchesTheCpuToFloatPrecision) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    gpu::VocabularyHead<float> head = makeHead(weights);
    LinearLayer<double> layer = makeReferenceLayer(weights);
    const std::vector<double> decoderOutput = pattern(6 * width, 1.0, 0.7);
    const std::vector<std::size_t> labels = { 0, 699, 12, 350, 12, 5 };

    std::vector<double> expectedDx;
    const double expectedLoss = referenceOutputLayer(layer, decoderOutput, labels, expectedDx);
    head.zeroGradients();
    std::vector<float> dx;
    const float loss = head.outputLayer(convert<float>(decoderOutput), labels, dx);

    EXPECT_NEAR(loss, expectedLoss, 1e-4 * std::max(1.0, std::fabs(expectedLoss)));
    ASSERT_EQ(dx.size(), expectedDx.size());
    for (std::size_t i = 0; i < dx.size(); i++) {
        ASSERT_NEAR(dx[i], expectedDx[i], 1e-4) << "index " << i;
    }
}

TEST(GpuVocabularyHeadTest, TrainingLowersTheLoss) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    gpu::VocabularyHead<float> head = makeHead(weights);
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

TEST(GpuVocabularyHeadTest, EmbeddingGradientsAreAppliedByTheOptimizer) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    gpu::VocabularyHead<float> head = makeHead(weights);
    const std::vector<std::size_t> ids = { 4, 9, 4 };
    const std::vector<double> rowGradients = pattern(ids.size() * width, 0.6, 1.1);

    head.zeroGradients();
    head.accumulateSourceGradient(ids, convert<float>(rowGradients));
    head.updateSgd(0.1f);

    std::vector<float> source, target, projection, bias;
    head.downloadWeights(source, target, projection, bias);
    const std::vector<double> expected = referenceTableUpdate(weights.source, sourceVocab, ids, rowGradients, 1, false, 0.1);
    for (std::size_t i = 0; i < expected.size(); i++) {
        ASSERT_NEAR(source[i], expected[i], 1e-5) << "index " << i;
    }
}

TEST(GpuVocabularyHeadTest, RejectsWhatTheUnderlyingHeadRejects) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    gpu::VocabularyHead<float> head = makeHead(weights);
    std::vector<float> rows;

    EXPECT_THROW(head.lookupSource({}, rows), InvalidSizeError);
    EXPECT_THROW(head.lookupSource({ sourceVocab }, rows), InvalidParameterError);
    EXPECT_THROW(head.updateSgd(0.0f), InvalidParameterError);
    EXPECT_THROW(head.updateAdam(0.01f, 0), InvalidParameterError);
    EXPECT_THROW(gpu::VocabularyHead<float>(0, 1, 1), InvalidParameterSizeError);
}

TEST(GpuVocabularyHeadTest, AHeadCanBeMoved) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    gpu::VocabularyHead<float> first = makeHead(weights);

    gpu::VocabularyHead<float> second = std::move(first);
    gpu::VocabularyHead<float> third(1, 1, 1);
    third = std::move(second);

    std::vector<float> source, target, projection, bias;
    third.downloadWeights(source, target, projection, bias);
    EXPECT_EQ(source, convert<float>(weights.source));
}
