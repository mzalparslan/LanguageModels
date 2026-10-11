#include "pch.h"
#include "GpuVocabularyHead.h"
#include "MiniTransformer.h"
#include "TestSupport.h"
#include <cmath>
#include <iostream>

// trainStepDevice() must be the same training step as trainStep(), to within rounding:
// it moves the vocabulary-sized work to the GPU, which sums in a different order, so
// the results are close but not bit-identical. These tests run in double precision so
// "close" can be held to a tight tolerance, which needs a GPU that computes in double
// (CUDA: Metal has no double; MiniTransformerFloatTests.cpp covers float). They report
// themselves skipped on a machine without one.
#define SKIP_WITHOUT_GPU() \
    if (!gpu::isAvailable<double>()) { \
        std::cout << "[  SKIPPED ] no GPU that computes in double (CUDA), or this build has none\n"; \
        return; \
    }

namespace {
    const std::size_t sourceVocab = 400;
    const std::size_t targetVocab = 3000;

    // Repeated words in both sequences: their embedding gradients must be added, in order.
    const std::vector<std::size_t> source = { 3, 4, 3, 6, 7, 4 };
    const std::vector<std::size_t> decoderInput = { 1, 2, 3, 2, 5, 6, 3 };
    const std::vector<std::size_t> expectedOutput = { 2, 3, 2, 5, 6, 3, 0 };

    // Two identical models: one trained with trainStep(), the other on the GPU, then compared.
    void expectSameTraining(bool adam, std::size_t steps, double lossTolerance, double weightTolerance) {
        MiniTransformer<double> cpu(sourceVocab, targetVocab);
        MiniTransformer<double> onDevice(sourceVocab, targetVocab);
        gpu::VocabularyHead<double> head = onDevice.createDeviceHead();

        for (std::size_t step = 1; step <= steps; step++) {
            const UpdateRule rule = adam ? UpdateRule::adam(step) : UpdateRule::sgd();
            const double cpuLoss = cpu.trainStep(source, decoderInput, expectedOutput, 0.005, rule);
            const double gpuLoss = onDevice.trainStepDevice(head, source, decoderInput, expectedOutput, 0.005, rule);

            ASSERT_NEAR(gpuLoss, cpuLoss, lossTolerance * std::max(1.0, std::fabs(cpuLoss))) << "loss, step " << step;
        }

        // Same weights <=> same logits, so compare the models through them.
        onDevice.downloadFromDeviceHead(head);
        for (const auto& input : { decoderInput, std::vector<std::size_t>{ 9, 8, 7 } }) {
            Tensor<double> cpuLogits, gpuLogits;
            cpu.forward(source, input, cpuLogits);
            onDevice.forward(source, input, gpuLogits);
            EXPECT_TRUE(testsupport::tensorsNear(gpuLogits, cpuLogits, weightTolerance));
        }
    }
}

TEST(MiniTransformerDeviceTest, MatchesTrainStepWithAdamToWithinRounding) {
    SKIP_WITHOUT_GPU();
    expectSameTraining(true, 5, 1e-9, 1e-8);
}

TEST(MiniTransformerDeviceTest, MatchesTrainStepWithPlainSgdToWithinRounding) {
    SKIP_WITHOUT_GPU();
    expectSameTraining(false, 5, 1e-9, 1e-8);
}

TEST(MiniTransformerDeviceTest, MatchesTrainStepOverManySteps) {
    SKIP_WITHOUT_GPU();
    // Rounding differences do not blow up over a longer run.
    expectSameTraining(true, 40, 1e-7, 1e-6);
}

TEST(MiniTransformerDeviceTest, TrainingOnTheGpuReducesTheLoss) {
    SKIP_WITHOUT_GPU();
    MiniTransformer<double> model(sourceVocab, targetVocab);
    gpu::VocabularyHead<double> head = model.createDeviceHead();

    double first = 0.0, last = 0.0;
    for (std::size_t step = 1; step <= 40; step++) {
        last = model.trainStepDevice(head, source, decoderInput, expectedOutput, 0.005, UpdateRule::adam(step));
        if (step == 1) {
            first = last;
        }
    }

    EXPECT_LT(last, 0.5 * first);
}

TEST(MiniTransformerDeviceTest, ATrainedModelTranslatesOnTheCpuAfterTheDownload) {
    SKIP_WITHOUT_GPU();
    MiniTransformer<double> model(12, 10);
    gpu::VocabularyHead<double> head = model.createDeviceHead();
    const std::vector<std::size_t> tinySource = { 3, 4, 5, 6 };
    const std::vector<std::size_t> tinyInput = { 1, 2, 3, 4, 5 };
    const std::vector<std::size_t> tinyOutput = { 2, 3, 4, 5, 0 };

    for (std::size_t step = 1; step <= 150; step++) {
        (void)model.trainStepDevice(head, tinySource, tinyInput, tinyOutput, 0.005, UpdateRule::adam(step));
    }
    model.downloadFromDeviceHead(head);

    // Start token 1, end token 0: the sentence the model was trained on.
    EXPECT_EQ(model.generate(tinySource, 1, 0, 10), (std::vector<std::size_t>{ 1, 2, 3, 4, 5, 0 }));
}

TEST(MiniTransformerDeviceTest, ALongSequenceWorks) {
    SKIP_WITHOUT_GPU();
    MiniTransformer<double> cpu(sourceVocab, targetVocab);
    MiniTransformer<double> onDevice(sourceVocab, targetVocab);
    gpu::VocabularyHead<double> head = onDevice.createDeviceHead();
    std::vector<std::size_t> longSource, longInput, longOutput;
    for (std::size_t i = 0; i < MiniTransformerConfig::maxSeqLen; i++) {
        longSource.push_back(1 + i % 50);
        longInput.push_back(1 + (i * 7) % 300);
        longOutput.push_back((i * 11) % 300);
    }

    const double cpuLoss = cpu.trainStep(longSource, longInput, longOutput, 0.005, UpdateRule::adam(1));
    const double gpuLoss = onDevice.trainStepDevice(head, longSource, longInput, longOutput, 0.005, UpdateRule::adam(1));

    EXPECT_NEAR(gpuLoss, cpuLoss, 1e-9 * cpuLoss);
}

TEST(MiniTransformerDeviceTest, ValidatesItsArgumentsLikeTrainStep) {
    SKIP_WITHOUT_GPU();
    MiniTransformer<double> model(sourceVocab, targetVocab);
    gpu::VocabularyHead<double> head = model.createDeviceHead();
    auto rule = UpdateRule::adam(1);

    EXPECT_THROW(model.trainStepDevice(head, source, decoderInput, expectedOutput, 0.0, rule), InvalidParameterError);
    EXPECT_THROW(model.trainStepDevice(head, source, decoderInput, expectedOutput, -1.0, rule), InvalidParameterError);
    EXPECT_THROW(model.trainStepDevice(head, source, decoderInput, expectedOutput, std::nan(""), rule), NaNError);
    // One label per decoder position.
    EXPECT_THROW(model.trainStepDevice(head, source, decoderInput, { 1, 2 }, 0.001, rule), InvalidSizeError);
    // Labels must be target-vocabulary ids.
    EXPECT_THROW(model.trainStepDevice(head, source, decoderInput, { 2, 3, 2, 5, 6, 3, targetVocab }, 0.001, rule),
        InvalidParameterError);
    // Inputs are validated too.
    EXPECT_THROW(model.trainStepDevice(head, {}, decoderInput, expectedOutput, 0.001, rule), InvalidSizeError);
    EXPECT_THROW(model.trainStepDevice(head, { sourceVocab }, decoderInput, expectedOutput, 0.001, rule),
        InvalidParameterError);
}

TEST(MiniTransformerDeviceTest, ARejectedStepLeavesTheHeadUsable) {
    SKIP_WITHOUT_GPU();
    MiniTransformer<double> model(sourceVocab, targetVocab);
    gpu::VocabularyHead<double> head = model.createDeviceHead();

    EXPECT_THROW(model.trainStepDevice(head, source, decoderInput, { 2, 3, 2, 5, 6, 3, targetVocab + 1 }, 0.001, UpdateRule::adam(1)),
        InvalidParameterError);

    EXPECT_NO_THROW(model.trainStepDevice(head, source, decoderInput, expectedOutput, 0.001, UpdateRule::adam(1)));
}

TEST(MiniTransformerDeviceTest, AHeadOfTheWrongSizeIsRejected) {
    SKIP_WITHOUT_GPU();
    MiniTransformer<double> model(sourceVocab, targetVocab);
    MiniTransformer<double> other(sourceVocab + 1, targetVocab);
    gpu::VocabularyHead<double> wrongHead = other.createDeviceHead();

    EXPECT_THROW(model.trainStepDevice(wrongHead, source, decoderInput, expectedOutput, 0.001, UpdateRule::adam(1)),
        InvalidSizeError);
    EXPECT_THROW(model.downloadFromDeviceHead(wrongHead), InvalidSizeError);
}

TEST(MiniTransformerDeviceTest, WithoutAGpuCreatingAHeadThrowsDeviceError) {
    if (gpu::isAvailable<double>()) {
        return; // only meaningful where there is nothing to use
    }
    MiniTransformer<double> model(sourceVocab, targetVocab);

    EXPECT_THROW((void)model.createDeviceHead(), DeviceError);
}
