#include "pch.h"
#include "CudaRuntime.h"
#include "MiniTransformerPipeline.h"
#include "Parameter.h"
#include "RMSNorm.h"
#include "RotaryEmbedding.h"
#include "TestSupport.h"
#include <cmath>
#include <iostream>

// The model works in float as well as double. float halves the memory traffic and is what
// a GPU is built for; it costs precision, so what is checked here is that nothing in the
// library quietly assumes double (a conversion, a literal), and that the float model trains,
// translates and gives the same results whichever way a step is run.
#define SKIP_WITHOUT_GPU() \
    if (!cuda::isAvailable()) { \
        std::cout << "[  SKIPPED ] no CUDA device, or this build has no CUDA support\n"; \
        return; \
    }

namespace {
    const std::size_t sourceVocab = 400;
    const std::size_t targetVocab = 3000;

    const std::vector<std::size_t> source = { 3, 4, 3, 6, 7, 4 };
    const std::vector<std::size_t> decoderInput = { 1, 2, 3, 2, 5, 6, 3 };
    const std::vector<std::size_t> expectedOutput = { 2, 3, 2, 5, 6, 3, 0 };

    Tensor<float> patternMatrix(std::size_t rows, std::size_t cols, float phase) {
        Tensor<float> tensor({ rows, cols });
        for (std::size_t i = 0; i < tensor.size(); i++) {
            tensor[i] = 0.5f * std::sin(0.7f * static_cast<float>(i) + phase) + 0.1f * static_cast<float>(i % 3);
        }
        return tensor;
    }

    Tensor<double> asDouble(const Tensor<float>& tensor) {
        Tensor<double> converted(tensor.shape);
        for (std::size_t i = 0; i < tensor.size(); i++) {
            converted[i] = static_cast<double>(tensor[i]);
        }
        return converted;
    }

    void expectClose(const Tensor<float>& actual, const Tensor<double>& expected, double tolerance, const char* what) {
        ASSERT_EQ(actual.shape, expected.shape) << what;
        for (std::size_t i = 0; i < actual.size(); i++) {
            ASSERT_NEAR(static_cast<double>(actual[i]), expected[i], tolerance) << what << ", index " << i;
        }
    }
}

// ---------------------------------------------------- the float layers agree with double

TEST(FloatLayersTest, RMSNormMatchesTheDoubleVersionForwardAndBackward) {
    RMSNorm<float> floatNorm(8);
    RMSNorm<double> doubleNorm(8);
    Tensor<float> x = patternMatrix(5, 8, 0.3f);
    Tensor<float> dout = patternMatrix(5, 8, 1.1f);
    Tensor<double> xd = asDouble(x), doutd = asDouble(dout);

    Tensor<float> y, dx;
    Tensor<double> yd, dxd;
    floatNorm.forward(x, y);
    floatNorm.backward(dout, dx);
    doubleNorm.forward(xd, yd);
    doubleNorm.backward(doutd, dxd);

    expectClose(y, yd, 1e-5, "RMSNorm output");
    expectClose(dx, dxd, 1e-5, "RMSNorm input gradient");
}

TEST(FloatLayersTest, RotaryEmbeddingMatchesTheDoubleVersionAndUndoesItself) {
    RotaryEmbedding<float, MiniTransformerConfig> floatRope;
    RotaryEmbedding<double, MiniTransformerConfig> doubleRope;
    Tensor<float> x = patternMatrix(6, MiniTransformerConfig::d_head, 0.2f);
    Tensor<double> xd = asDouble(x);
    const Tensor<float> original = x;

    floatRope.apply(x);
    doubleRope.apply(xd);
    expectClose(x, xd, 1e-5, "rotated values");

    floatRope.applyInverse(x);
    expectClose(x, asDouble(original), 1e-5, "values after rotating back");
}

TEST(FloatLayersTest, AdamAndSgdUpdatesOfAFloatParameterAreRight) {
    RandomEngine rng(1);
    Parameter<float> adam;
    adam.init({ 1 }, 1.0f, rng);
    adam.value[0] = 1.0f;
    adam.grad[0] = 0.5f;
    Parameter<float> sgd = adam;

    adam.update(0.1f, UpdateRule::adam(1));
    sgd.update(0.1f, UpdateRule::sgd());

    // Adam's first step moves a weight by about lr, whatever the size of the gradient:
    // the bias-corrected moments are g and g^2, so the step is lr * g / (|g| + eps).
    EXPECT_NEAR(adam.value[0], 0.9f, 1e-6f);
    EXPECT_NEAR(sgd.value[0], 1.0f - 0.1f * 0.5f, 1e-7f);
}

// ------------------------------------------------------------------ the float model

TEST(MiniTransformerFloatTest, ForwardGivesFiniteLogitsOfTheRightShape) {
    MiniTransformer<float> model(12, 10);
    Tensor<float> logits;

    model.forward({ 3, 4, 5, 6 }, { 1, 2, 3, 4, 5 }, logits);

    EXPECT_EQ(logits.shape, (std::vector<std::size_t>{ 5, 10 }));
    for (std::size_t i = 0; i < logits.size(); i++) {
        ASSERT_TRUE(std::isfinite(logits[i])) << i;
    }
}

TEST(MiniTransformerFloatTest, TrainingReducesTheLossWithAdamAndWithSgd) {
    for (bool adam : { true, false }) {
        MiniTransformer<float> model(12, 10);
        float first = 0.0f, last = 0.0f;
        for (std::size_t step = 1; step <= 100; step++) {
            last = model.trainStep({ 3, 4, 5, 6 }, { 1, 2, 3, 4, 5 }, { 2, 3, 4, 5, 0 }, adam ? 0.005f : 0.02f,
                adam ? UpdateRule::adam(step) : UpdateRule::sgd());
            if (step == 1) {
                first = last;
            }
        }
        EXPECT_LT(last, 0.2f * first) << (adam ? "Adam" : "SGD");
    }
}

TEST(MiniTransformerFloatTest, ATrainedModelTranslatesItsSentence) {
    MiniTransformer<float> model(12, 10);
    const std::vector<std::size_t> tinySource = { 3, 4, 5, 6 };
    for (std::size_t step = 1; step <= 150; step++) {
        (void)model.trainStep(tinySource, { 1, 2, 3, 4, 5 }, { 2, 3, 4, 5, 0 }, 0.005f, UpdateRule::adam(step));
    }

    EXPECT_EQ(model.generate(tinySource, 1, 0, 10), (std::vector<std::size_t>{ 1, 2, 3, 4, 5, 0 }));
    auto best = model.beamSearch(tinySource, 1, 0, 10, 3);
    EXPECT_EQ(best.tokens, (std::vector<std::size_t>{ 1, 2, 3, 4, 5, 0 }));
    EXPECT_LE(best.logProbability, 0.0f);
}

TEST(MiniTransformerFloatTest, TheThreadedStepIsBitIdenticalToTrainStepInFloatToo) {
    for (std::size_t threads : { 1u, 2u, 8u, 0u }) {
        MiniTransformer<float> sequential(sourceVocab, targetVocab);
        MiniTransformer<float> threaded(sourceVocab, targetVocab);

        for (std::size_t step = 1; step <= 4; step++) {
            const float a = sequential.trainStep(source, decoderInput, expectedOutput, 0.005f, UpdateRule::adam(step));
            const float b = threaded.trainStepMultipleThread(source, decoderInput, expectedOutput, 0.005f,
                UpdateRule::adam(step), threads);
            ASSERT_EQ(a, b) << "loss, step " << step << ", threads " << threads;
        }

        Tensor<float> logitsA, logitsB;
        sequential.forward(source, decoderInput, logitsA);
        threaded.forward(source, decoderInput, logitsB);
        EXPECT_EQ(logitsA.data, logitsB.data) << "threads " << threads;
    }
}

TEST(MiniTransformerFloatTest, RejectsTheSameInvalidInputAsTheDoubleModel) {
    MiniTransformer<float> model(sourceVocab, targetVocab);
    auto rule = UpdateRule::adam(1);

    EXPECT_THROW(model.trainStep(source, decoderInput, expectedOutput, 0.0f, rule), InvalidParameterError);
    EXPECT_THROW(model.trainStep(source, decoderInput, expectedOutput, std::nanf(""), rule), NaNError);
    EXPECT_THROW(model.trainStep(source, decoderInput, { 1, 2 }, 0.001f, rule), InvalidSizeError);
    EXPECT_THROW(model.trainStep(source, decoderInput, { 2, 3, 2, 5, 6, 3, targetVocab }, 0.001f, rule),
        InvalidParameterError);
}

// ---------------------------------------------------------------- the float GPU step

TEST(MiniTransformerFloatCudaTest, MatchesTheCpuFloatStepToWithinFloatPrecision) {
    SKIP_WITHOUT_GPU();
    MiniTransformer<float> cpu(sourceVocab, targetVocab);
    MiniTransformer<float> gpu(sourceVocab, targetVocab);
    cuda::VocabularyHead<float> head = gpu.createCudaHead();

    for (std::size_t step = 1; step <= 5; step++) {
        const float cpuLoss = cpu.trainStep(source, decoderInput, expectedOutput, 0.005f, UpdateRule::adam(step));
        const float gpuLoss = gpu.trainStepCuda(head, source, decoderInput, expectedOutput, 0.005f, UpdateRule::adam(step));
        ASSERT_NEAR(gpuLoss, cpuLoss, 1e-4f * std::max(1.0f, std::fabs(cpuLoss))) << "loss, step " << step;
    }

    gpu.downloadFromCudaHead(head);
    Tensor<float> cpuLogits, gpuLogits;
    cpu.forward(source, decoderInput, cpuLogits);
    gpu.forward(source, decoderInput, gpuLogits);
    expectClose(gpuLogits, asDouble(cpuLogits), 1e-2, "logits after training");
}

TEST(MiniTransformerFloatCudaTest, TrainingOnTheGpuInFloatReducesTheLossAndTranslates) {
    SKIP_WITHOUT_GPU();
    MiniTransformer<float> model(12, 10);
    cuda::VocabularyHead<float> head = model.createCudaHead();
    const std::vector<std::size_t> tinySource = { 3, 4, 5, 6 };

    float first = 0.0f, last = 0.0f;
    for (std::size_t step = 1; step <= 150; step++) {
        last = model.trainStepCuda(head, tinySource, { 1, 2, 3, 4, 5 }, { 2, 3, 4, 5, 0 }, 0.005f, UpdateRule::adam(step));
        if (step == 1) {
            first = last;
        }
    }
    model.downloadFromCudaHead(head);

    EXPECT_LT(last, 0.1f * first);
    EXPECT_EQ(model.generate(tinySource, 1, 0, 10), (std::vector<std::size_t>{ 1, 2, 3, 4, 5, 0 }));
}

// -------------------------------------------------------------------- the pipeline

namespace {
    using FloatPipeline = Pipeline<float, MiniTransformer<float>, TranslationParameters<float>>;

    Logger& quietLogger() {
        static Logger logger(LogLevel::Critical);
        return logger;
    }

    std::vector<SentencePair> toyPairs() {
        return {
            { "hello there", "bonjour la" },
            { "good morning", "bon matin" },
            { "thank you", "merci vous" },
            { "see you", "voir vous" },
        };
    }

    TranslationParameters<float> fastParameters(std::size_t epochs) {
        TranslationParameters<float> parameters;
        parameters.epochs = epochs;
        parameters.learningRate = 0.005f;
        parameters.logEveryEpochs = 0;
        return parameters;
    }
}

TEST(FloatTranslationPipelineTest, TrainsAModelThatTranslatesItsTrainingSentences) {
    FloatPipeline pipeline(fastParameters(300), ExecutionStrategy::Sequential, quietLogger());

    pipeline.train(toyPairs());

    EXPECT_EQ(pipeline.modelAdapter().translate("hello there"), "bonjour la");
    EXPECT_DOUBLE_EQ(pipeline.evaluate(toyPairs()).exactMatchRate, 1.0);
}

TEST(FloatTranslationPipelineTest, ParallelExecutionGivesTheSameResultsAsSequential) {
    FloatPipeline sequential(fastParameters(20), ExecutionStrategy::Sequential, quietLogger());
    FloatPipeline parallel(fastParameters(20), ExecutionStrategy::Parallel, quietLogger());

    sequential.train(toyPairs());
    parallel.train(toyPairs());

    EXPECT_EQ(sequential.modelAdapter().finalTrainingLoss(), parallel.modelAdapter().finalTrainingLoss());
    EXPECT_EQ(sequential.evaluate(toyPairs()).teacherForced.loss, parallel.evaluate(toyPairs()).teacherForced.loss);
}

TEST(FloatTranslationPipelineTest, CudaExecutionTrainsAModelThatTranslates) {
    SKIP_WITHOUT_GPU();
    FloatPipeline pipeline(fastParameters(300), ExecutionStrategy::Cuda, quietLogger());

    pipeline.train(toyPairs());

    EXPECT_EQ(pipeline.modelAdapter().translate("thank you"), "merci vous");
    EXPECT_DOUBLE_EQ(pipeline.evaluate(toyPairs()).exactMatchRate, 1.0);
}
