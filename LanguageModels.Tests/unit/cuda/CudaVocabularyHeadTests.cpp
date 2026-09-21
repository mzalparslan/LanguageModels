#include "pch.h"
#include "CudaRuntime.h"
#include "CudaVocabularyHead.h"
#include "LinearLayer.h"
#include "Parameter.h"
#include <cmath>
#include <iostream>
#include <limits>

// The GPU head is checked against the CPU code it replaces, in double precision, where
// the two must agree to within rounding (the GPU sums in a different order and fuses
// multiply-adds, so they are not bit-identical). Tests that need a GPU report themselves
// skipped on a machine without one.
#define SKIP_WITHOUT_GPU() \
    if (!cuda::isAvailable()) { \
        std::cout << "[  SKIPPED ] no CUDA device, or this build has no CUDA support\n"; \
        return; \
    }

namespace {
    const std::size_t sourceVocab = 50;
    const std::size_t targetVocab = 700;   // large enough for several blocks per row
    const std::size_t width = 8;

    // Deterministic values of a chosen scale.
    std::vector<double> pattern(std::size_t count, double scale, double phase) {
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; i++) {
            values[i] = scale * std::sin(0.37 * static_cast<double>(i) + phase);
        }
        return values;
    }

    template <typename T>
    std::vector<T> convert(const std::vector<double>& values) {
        std::vector<T> converted;
        converted.reserve(values.size());
        for (double value : values) {
            converted.push_back(static_cast<T>(value));
        }
        return converted;
    }

    // The four parameters of a head, with some starting values.
    class Weights {
    public:
        std::vector<double> source = pattern(sourceVocab * width, 0.5, 0.1);
        std::vector<double> target = pattern(targetVocab * width, 0.5, 0.2);
        std::vector<double> projection = pattern(width * targetVocab, 0.3, 0.3);
        std::vector<double> bias = pattern(targetVocab, 0.1, 0.4);
    };

    cuda::VocabularyHead<double> makeHead(const Weights& weights) {
        cuda::VocabularyHead<double> head(sourceVocab, targetVocab, width);
        head.uploadWeights(weights.source, weights.target, weights.projection, weights.bias);
        return head;
    }

    // The CPU's output layer: LinearLayer, then softmax + cross-entropy exactly as
    // MiniTransformer::trainStep() does it. Fills dx and leaves the gradients in layer.
    double referenceOutputLayer(LinearLayer<double>& layer, const std::vector<double>& x,
        const std::vector<std::size_t>& labels, std::vector<double>& dx) {
        const std::size_t rows = labels.size();
        Tensor<double> input({ rows, width });
        input.data = x;
        Tensor<double> logits;
        layer.forward(input, logits);

        Tensor<double> dlogits(logits.shape, 0);
        double loss = 0;
        for (std::size_t i = 0; i < rows; i++) {
            double maxValue = -1e9;
            for (std::size_t j = 0; j < targetVocab; j++) {
                maxValue = std::max(maxValue, logits.data[i * targetVocab + j]);
            }
            double sum = 0;
            std::vector<double> probs(targetVocab);
            for (std::size_t j = 0; j < targetVocab; j++) {
                probs[j] = std::exp(logits.data[i * targetVocab + j] - maxValue);
                sum += probs[j];
            }
            for (std::size_t j = 0; j < targetVocab; j++) {
                probs[j] /= sum;
            }
            loss -= std::log(probs[labels[i]]);
            for (std::size_t j = 0; j < targetVocab; j++) {
                dlogits.data[i * targetVocab + j] = probs[j];
            }
            dlogits.data[i * targetVocab + labels[i]] -= 1.0;
        }

        Tensor<double> dxTensor;
        layer.backward(dlogits, dxTensor);
        dx = dxTensor.data;
        return loss;
    }

    LinearLayer<double> makeReferenceLayer(const Weights& weights) {
        RandomEngine rng(1);
        LinearLayer<double> layer(width, targetVocab, rng);
        layer.W.value.data = weights.projection;
        layer.b.value.data = weights.bias;
        layer.zeroGrad();
        return layer;
    }

    void expectNear(const std::vector<double>& actual, const std::vector<double>& expected, double tolerance, const char* what) {
        ASSERT_EQ(actual.size(), expected.size()) << what;
        for (std::size_t i = 0; i < actual.size(); i++) {
            ASSERT_NEAR(actual[i], expected[i], tolerance) << what << ", index " << i;
        }
    }

    void downloadAll(const cuda::VocabularyHead<double>& head, Weights& out) {
        head.downloadWeights(out.source, out.target, out.projection, out.bias);
    }
}

// -------------------------------------------------------- weights and lookups

TEST(CudaVocabularyHeadTest, DownloadReturnsExactlyWhatWasUploaded) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);

    Weights back;
    downloadAll(head, back);

    EXPECT_EQ(back.source, weights.source);
    EXPECT_EQ(back.target, weights.target);
    EXPECT_EQ(back.projection, weights.projection);
    EXPECT_EQ(back.bias, weights.bias);
}

TEST(CudaVocabularyHeadTest, ReportsItsSizes) {
    SKIP_WITHOUT_GPU();
    cuda::VocabularyHead<double> head(sourceVocab, targetVocab, width);

    EXPECT_EQ(head.sourceVocab(), sourceVocab);
    EXPECT_EQ(head.targetVocab(), targetVocab);
    EXPECT_EQ(head.width(), width);
}

TEST(CudaVocabularyHeadTest, LookupsReturnTheRowsOfTheTables) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    const std::vector<std::size_t> sourceIds = { 3, 0, 49, 3 };
    const std::vector<std::size_t> targetIds = { 699, 5, 5, 0, 350 };

    std::vector<double> sourceRows, targetRows;
    head.lookupSource(sourceIds, sourceRows);
    head.lookupTarget(targetIds, targetRows);

    ASSERT_EQ(sourceRows.size(), sourceIds.size() * width);
    ASSERT_EQ(targetRows.size(), targetIds.size() * width);
    for (std::size_t r = 0; r < sourceIds.size(); r++) {
        for (std::size_t c = 0; c < width; c++) {
            EXPECT_EQ(sourceRows[r * width + c], weights.source[sourceIds[r] * width + c]);
        }
    }
    for (std::size_t r = 0; r < targetIds.size(); r++) {
        for (std::size_t c = 0; c < width; c++) {
            EXPECT_EQ(targetRows[r * width + c], weights.target[targetIds[r] * width + c]);
        }
    }
}

// --------------------------------------------------------------- output layer

TEST(CudaVocabularyHeadTest, OutputLayerMatchesTheCpuLossAndInputGradient) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    LinearLayer<double> layer = makeReferenceLayer(weights);
    const std::vector<double> decoderOutput = pattern(6 * width, 1.0, 0.7);
    const std::vector<std::size_t> labels = { 0, 699, 12, 350, 12, 5 };

    std::vector<double> expectedDx;
    const double expectedLoss = referenceOutputLayer(layer, decoderOutput, labels, expectedDx);
    head.zeroGradients();
    std::vector<double> dx;
    const double loss = head.outputLayer(decoderOutput, labels, dx);

    EXPECT_NEAR(loss, expectedLoss, 1e-10 * std::max(1.0, std::fabs(expectedLoss)));
    expectNear(dx, expectedDx, 1e-11, "decoder output gradient");
}

TEST(CudaVocabularyHeadTest, PlainGradientDescentMatchesTheCpuProjectionUpdate) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    LinearLayer<double> layer = makeReferenceLayer(weights);
    const std::vector<double> decoderOutput = pattern(5 * width, 1.0, 0.2);
    const std::vector<std::size_t> labels = { 1, 2, 3, 4, 5 };

    std::vector<double> ignored;
    referenceOutputLayer(layer, decoderOutput, labels, ignored);
    layer.update(0.05, UpdateRule::sgd());
    head.zeroGradients();
    head.outputLayer(decoderOutput, labels, ignored);
    head.updateSgd(0.05);

    Weights after;
    downloadAll(head, after);
    expectNear(after.projection, layer.W.value.data, 1e-12, "projection weights");
    expectNear(after.bias, layer.b.value.data, 1e-12, "projection bias");
    // The embedding tables had no gradient, so they did not move.
    EXPECT_EQ(after.source, weights.source);
    EXPECT_EQ(after.target, weights.target);
}

TEST(CudaVocabularyHeadTest, AdamOnTheProjectionMatchesTheCpuOverSeveralSteps) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    LinearLayer<double> layer = makeReferenceLayer(weights);
    const std::vector<double> decoderOutput = pattern(4 * width, 1.0, 0.9);
    const std::vector<std::size_t> labels = { 10, 20, 30, 40 };

    for (std::size_t step = 1; step <= 4; step++) {
        std::vector<double> ignored;
        layer.zeroGrad();
        referenceOutputLayer(layer, decoderOutput, labels, ignored);
        layer.update(0.01, UpdateRule::adam(step));

        head.zeroGradients();
        head.outputLayer(decoderOutput, labels, ignored);
        head.updateAdam(0.01, step);
    }

    Weights after;
    downloadAll(head, after);
    expectNear(after.projection, layer.W.value.data, 1e-11, "projection weights");
    expectNear(after.bias, layer.b.value.data, 1e-11, "projection bias");
}

TEST(CudaVocabularyHeadTest, TrainingOnTheGpuLowersTheLoss) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    const std::vector<double> decoderOutput = pattern(5 * width, 1.0, 0.5);
    const std::vector<std::size_t> labels = { 7, 8, 9, 10, 11 };

    double first = 0.0, last = 0.0;
    for (std::size_t step = 1; step <= 30; step++) {
        std::vector<double> dx;
        head.zeroGradients();
        last = head.outputLayer(decoderOutput, labels, dx);
        head.updateAdam(0.05, step);
        if (step == 1) {
            first = last;
        }
    }

    EXPECT_LT(last, 0.2 * first);
}

// ------------------------------------------------------------ embedding tables

namespace {
    // The CPU update of one embedding table given per-row gradients, for comparison.
    std::vector<double> referenceTableUpdate(const std::vector<double>& table, std::size_t vocab,
        const std::vector<std::size_t>& ids, const std::vector<double>& rowGradients,
        std::size_t steps, bool adam, double learningRate) {
        Parameter<double> parameter;
        RandomEngine rng(1);
        parameter.init({ vocab, width }, 0.1, rng);
        parameter.value.data = table;

        for (std::size_t step = 1; step <= steps; step++) {
            parameter.zeroGrad();
            for (std::size_t r = 0; r < ids.size(); r++) {
                for (std::size_t c = 0; c < width; c++) {
                    parameter.grad.data[ids[r] * width + c] += rowGradients[r * width + c];
                }
            }
            parameter.update(learningRate, adam ? UpdateRule::adam(step) : UpdateRule::sgd());
        }
        return parameter.value.data;
    }
}

TEST(CudaVocabularyHeadTest, EmbeddingGradientsAreAddedRowByRowIncludingRepeatedWords) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    // Word 4 occurs three times, and the gradients are large enough to be clipped.
    const std::vector<std::size_t> ids = { 4, 9, 4, 0, 4 };
    const std::vector<double> rowGradients = pattern(ids.size() * width, 0.6, 1.1);

    head.zeroGradients();
    head.accumulateSourceGradient(ids, rowGradients);
    head.updateSgd(0.1);

    Weights after;
    downloadAll(head, after);
    expectNear(after.source, referenceTableUpdate(weights.source, sourceVocab, ids, rowGradients, 1, false, 0.1),
        1e-14, "source table");
    // Only the touched rows changed.
    for (std::size_t i = 0; i < weights.source.size(); i++) {
        const std::size_t row = i / width;
        if (row != 4 && row != 9 && row != 0) {
            ASSERT_EQ(after.source[i], weights.source[i]) << "index " << i;
        }
    }
}

TEST(CudaVocabularyHeadTest, AdamOnTheEmbeddingTablesMatchesTheCpuOverSeveralSteps) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    const std::vector<std::size_t> sourceIds = { 2, 7, 2, 30 };
    const std::vector<std::size_t> targetIds = { 100, 5, 699, 5, 0 };
    const std::vector<double> sourceGradients = pattern(sourceIds.size() * width, 2.0, 0.3);
    const std::vector<double> targetGradients = pattern(targetIds.size() * width, 0.02, 0.6);

    for (std::size_t step = 1; step <= 5; step++) {
        head.zeroGradients();
        head.accumulateSourceGradient(sourceIds, sourceGradients);
        head.accumulateTargetGradient(targetIds, targetGradients);
        head.updateAdam(0.01, step);
    }

    Weights after;
    downloadAll(head, after);
    expectNear(after.source, referenceTableUpdate(weights.source, sourceVocab, sourceIds, sourceGradients, 5, true, 0.01),
        1e-12, "source table");
    expectNear(after.target, referenceTableUpdate(weights.target, targetVocab, targetIds, targetGradients, 5, true, 0.01),
        1e-12, "target table");
}

TEST(CudaVocabularyHeadTest, UploadingAgainStartsAdamFromScratch) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    const std::vector<std::size_t> ids = { 1, 2 };
    const std::vector<double> gradients = pattern(ids.size() * width, 0.5, 0.2);

    // Two steps, then start over from the same weights: the first step of the
    // second run must equal the first step of the first run (fresh moments).
    for (std::size_t step = 1; step <= 2; step++) {
        head.zeroGradients();
        head.accumulateSourceGradient(ids, gradients);
        head.updateAdam(0.01, step);
    }
    head.uploadWeights(weights.source, weights.target, weights.projection, weights.bias);
    head.zeroGradients();
    head.accumulateSourceGradient(ids, gradients);
    head.updateAdam(0.01, 1);

    Weights after;
    downloadAll(head, after);
    expectNear(after.source, referenceTableUpdate(weights.source, sourceVocab, ids, gradients, 1, true, 0.01),
        1e-14, "source table");
}

// ------------------------------------------------------------------ float head

TEST(CudaVocabularyHeadTest, AFloatHeadAgreesWithTheDoubleHeadToFloatPrecision) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> doubleHead = makeHead(weights);
    cuda::VocabularyHead<float> floatHead(sourceVocab, targetVocab, width);
    floatHead.uploadWeights(convert<float>(weights.source), convert<float>(weights.target),
        convert<float>(weights.projection), convert<float>(weights.bias));
    const std::vector<double> decoderOutput = pattern(5 * width, 1.0, 0.4);
    const std::vector<std::size_t> labels = { 3, 30, 300, 600, 9 };

    std::vector<double> doubleDx;
    std::vector<float> floatDx;
    doubleHead.zeroGradients();
    floatHead.zeroGradients();
    const double doubleLoss = doubleHead.outputLayer(decoderOutput, labels, doubleDx);
    const float floatLoss = floatHead.outputLayer(convert<float>(decoderOutput), labels, floatDx);

    EXPECT_NEAR(floatLoss, doubleLoss, 1e-4 * doubleLoss);
    ASSERT_EQ(floatDx.size(), doubleDx.size());
    for (std::size_t i = 0; i < doubleDx.size(); i++) {
        ASSERT_NEAR(floatDx[i], doubleDx[i], 1e-4) << "index " << i;
    }
}

TEST(CudaVocabularyHeadTest, AFloatHeadTrains) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<float> head(sourceVocab, targetVocab, width);
    head.uploadWeights(convert<float>(weights.source), convert<float>(weights.target),
        convert<float>(weights.projection), convert<float>(weights.bias));
    const std::vector<float> decoderOutput = convert<float>(pattern(4 * width, 1.0, 0.8));
    const std::vector<std::size_t> labels = { 1, 2, 3, 4 };

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

// ---------------------------------------------------------------------- errors

TEST(CudaVocabularyHeadTest, RejectsInvalidSizesAndIds) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    std::vector<double> rows, dx;

    EXPECT_THROW(cuda::VocabularyHead<double>(0, 1, 1), InvalidParameterSizeError);
    EXPECT_THROW(cuda::VocabularyHead<double>(1, 0, 1), InvalidParameterSizeError);
    EXPECT_THROW(cuda::VocabularyHead<double>(1, 1, 0), InvalidParameterSizeError);

    EXPECT_THROW(head.uploadWeights({}, weights.target, weights.projection, weights.bias), InvalidSizeError);
    EXPECT_THROW(head.uploadWeights(weights.source, weights.target, weights.projection, {}), InvalidSizeError);

    EXPECT_THROW(head.lookupSource({}, rows), InvalidSizeError);
    EXPECT_THROW(head.lookupSource({ sourceVocab }, rows), InvalidParameterError);
    EXPECT_THROW(head.lookupTarget({ targetVocab }, rows), InvalidParameterError);

    const std::vector<double> output = pattern(2 * width, 1.0, 0.0);
    EXPECT_THROW(head.outputLayer(output, {}, dx), InvalidSizeError);
    EXPECT_THROW(head.outputLayer(output, { 1, 2, 3 }, dx), InvalidSizeError);
    EXPECT_THROW(head.outputLayer(output, { 1, targetVocab }, dx), InvalidParameterError);

    EXPECT_THROW(head.accumulateSourceGradient({ 1, 2 }, std::vector<double>(3, 0.0)), InvalidSizeError);
    EXPECT_THROW(head.accumulateTargetGradient({ targetVocab }, std::vector<double>(width, 0.0)), InvalidParameterError);
}

TEST(CudaVocabularyHeadTest, RejectsInvalidOptimizerArguments) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);

    EXPECT_THROW(head.updateSgd(0.0), InvalidParameterError);
    EXPECT_THROW(head.updateSgd(-1.0), InvalidParameterError);
    EXPECT_THROW(head.updateAdam(0.0, 1), InvalidParameterError);
    EXPECT_THROW(head.updateAdam(0.01, 0), InvalidParameterError);
    EXPECT_THROW(head.updateAdam(std::nan(""), 1), NaNError);
}

TEST(CudaVocabularyHeadTest, ALossThatIsNotFiniteIsReported) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    std::vector<double> output = pattern(2 * width, 1.0, 0.0);
    output[3] = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> dx;

    EXPECT_THROW(head.outputLayer(output, { 1, 2 }, dx), NaNError);
}

TEST(CudaVocabularyHeadTest, AGradientThatIsNotFiniteIsReportedByTheUpdate) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> head = makeHead(weights);
    std::vector<double> gradients = pattern(width, 0.5, 0.0);

    gradients[2] = std::numeric_limits<double>::quiet_NaN();
    head.zeroGradients();
    head.accumulateSourceGradient({ 3 }, gradients);
    EXPECT_THROW(head.updateAdam(0.01, 1), NaNError);
    head.zeroGradients();
    head.accumulateSourceGradient({ 3 }, gradients);
    EXPECT_THROW(head.updateSgd(0.01), NaNError);

    gradients[2] = std::numeric_limits<double>::infinity();
    head.zeroGradients();
    head.accumulateTargetGradient({ 3 }, gradients);
    EXPECT_THROW(head.updateAdam(0.01, 1), NonFiniteError);

    // The NaN is now in the weights and in Adam's moments, exactly as it would be on the CPU,
    // so every later update reports it again until the weights are uploaded afresh.
    head.zeroGradients();
    EXPECT_THROW(head.updateAdam(0.01, 2), NonFiniteError);
    head.uploadWeights(weights.source, weights.target, weights.projection, weights.bias);
    head.zeroGradients();
    EXPECT_NO_THROW(head.updateAdam(0.01, 1));
}

// ------------------------------------------------------------------ ownership

TEST(CudaVocabularyHeadTest, AHeadCanBeMoved) {
    SKIP_WITHOUT_GPU();
    Weights weights;
    cuda::VocabularyHead<double> first = makeHead(weights);

    cuda::VocabularyHead<double> second = std::move(first);
    cuda::VocabularyHead<double> third(1, 1, 1);
    third = std::move(second);

    Weights back;
    downloadAll(third, back);
    EXPECT_EQ(back.source, weights.source);
}

TEST(CudaVocabularyHeadTest, WithoutCudaAHeadCannotBeCreated) {
    if (cuda::isAvailable()) {
        return; // only meaningful where there is nothing to use
    }

    EXPECT_THROW(cuda::VocabularyHead<double>(1, 1, 1), CudaError);
    EXPECT_THROW(cuda::VocabularyHead<float>(1, 1, 1), CudaError);
}
