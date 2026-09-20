#include "pch.h"
#include "MiniTransformer.h"
#include "TestSupport.h"
#include <cmath>
#include <limits>

// trainStepMultipleThread() must be the same training step as trainStep(),
// only faster: same loss and same weights, bit for bit, whatever the thread
// count. Vocabularies are large enough that the threaded parts (projection,
// softmax, embedding tables) are really split into several chunks.

namespace {
    const std::size_t sourceVocab = 400;
    const std::size_t targetVocab = 3000;

    const std::vector<std::size_t> source = { 3, 4, 5, 6, 7, 8 };
    const std::vector<std::size_t> decoderInput = { 1, 2, 3, 4, 5, 6, 7 };
    const std::vector<std::size_t> expectedOutput = { 2, 3, 4, 5, 6, 7, 0 };

    // Two identical models: one trained with trainStep(), the other with
    // trainStepMultipleThread(), then compared exactly.
    void expectSameTraining(std::size_t threads, bool adam) {
        MiniTransformer<double> sequential(sourceVocab, targetVocab);
        MiniTransformer<double> threaded(sourceVocab, targetVocab);

        for (std::size_t step = 1; step <= 4; step++) {
            const UpdateRule rule = adam ? UpdateRule::adam(step) : UpdateRule::sgd();
            const double lossA = sequential.trainStep(source, decoderInput, expectedOutput, 0.005, rule);
            const double lossB = threaded.trainStepMultipleThread(source, decoderInput, expectedOutput, 0.005, rule, threads);

            ASSERT_EQ(lossA, lossB) << "loss, step " << step << ", threads " << threads;
        }

        // Same weights <=> same logits (checked over several inputs).
        for (const auto& input : { decoderInput, std::vector<std::size_t>{ 9, 8, 7 } }) {
            Tensor<double> logitsA, logitsB;
            sequential.forward(source, input, logitsA);
            threaded.forward(source, input, logitsB);
            EXPECT_TRUE(testsupport::tensorsEqual(logitsA, logitsB)) << "threads " << threads;
        }
    }
}

TEST(MiniTransformerThreadingTest, MatchesTrainStepExactlyWithAdamAtAnyThreadCount) {
    for (std::size_t threads : { 1u, 2u, 3u, 8u, 0u }) {
        expectSameTraining(threads, true);
    }
}

TEST(MiniTransformerThreadingTest, MatchesTrainStepExactlyWithPlainSgd) {
    for (std::size_t threads : { 1u, 4u }) {
        expectSameTraining(threads, false);
    }
}

TEST(MiniTransformerThreadingTest, SequentialAndThreadedStepsCanBeInterleaved) {
    MiniTransformer<double> reference(sourceVocab, targetVocab);
    MiniTransformer<double> mixed(sourceVocab, targetVocab);

    for (std::size_t step = 1; step <= 4; step++) {
        const UpdateRule rule = UpdateRule::adam(step);
        (void)reference.trainStep(source, decoderInput, expectedOutput, 0.005, rule);
        if (step % 2 == 0) {
            (void)mixed.trainStepMultipleThread(source, decoderInput, expectedOutput, 0.005, rule, 4);
        }
        else {
            (void)mixed.trainStep(source, decoderInput, expectedOutput, 0.005, rule);
        }
    }

    Tensor<double> a, b;
    reference.forward(source, decoderInput, a);
    mixed.forward(source, decoderInput, b);
    EXPECT_TRUE(testsupport::tensorsEqual(a, b));
}

TEST(MiniTransformerThreadingTest, SmallVocabulariesWorkToo) {
    MiniTransformer<double> sequential(12, 10);
    MiniTransformer<double> threaded(12, 10);
    const std::vector<std::size_t> input = { 1, 2, 3, 4, 5 };
    const std::vector<std::size_t> label = { 2, 3, 4, 5, 0 };

    for (std::size_t step = 1; step <= 3; step++) {
        const double lossA = sequential.trainStep(source, input, label, 0.005, UpdateRule::adam(step));
        const double lossB = threaded.trainStepMultipleThread(source, input, label, 0.005, UpdateRule::adam(step), 8);
        ASSERT_EQ(lossA, lossB);
    }
}

TEST(MiniTransformerThreadingTest, TrainingReducesTheLoss) {
    MiniTransformer<double> model(sourceVocab, targetVocab);

    double first = 0.0, last = 0.0;
    for (std::size_t step = 1; step <= 40; step++) {
        last = model.trainStepMultipleThread(source, decoderInput, expectedOutput, 0.005, UpdateRule::adam(step));
        if (step == 1) {
            first = last;
        }
    }

    EXPECT_LT(last, 0.5 * first);
}

TEST(MiniTransformerThreadingTest, ValidatesItsArgumentsLikeTrainStep) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    auto rule = UpdateRule::adam(1);

    EXPECT_THROW(model.trainStepMultipleThread(source, decoderInput, expectedOutput, 0.0, rule), InvalidParameterError);
    EXPECT_THROW(model.trainStepMultipleThread(source, decoderInput, expectedOutput, -1.0, rule), InvalidParameterError);
    EXPECT_THROW(model.trainStepMultipleThread(source, decoderInput, expectedOutput, std::nan(""), rule), NaNError);
    // One label per decoder position.
    EXPECT_THROW(model.trainStepMultipleThread(source, decoderInput, { 1, 2 }, 0.001, rule), InvalidSizeError);
    // Labels must be target-vocabulary ids.
    EXPECT_THROW(model.trainStepMultipleThread(source, decoderInput, { 2, 3, 4, 5, 6, 7, targetVocab }, 0.001, rule),
        InvalidParameterError);
    // Inputs are validated by forward().
    EXPECT_THROW(model.trainStepMultipleThread({}, decoderInput, expectedOutput, 0.001, rule), InvalidSizeError);
    EXPECT_THROW(model.trainStepMultipleThread({ sourceVocab }, decoderInput, expectedOutput, 0.001, rule),
        InvalidParameterError);
}

TEST(MiniTransformerThreadingTest, AnErrorFoundInsideTheThreadsReachesTheCaller) {
    MiniTransformer<double> model(sourceVocab, targetVocab);
    // A bad label is found while the threads are computing the softmax: the
    // error must reach the caller, and the pool must still work afterwards.
    EXPECT_THROW(model.trainStepMultipleThread(source, decoderInput, { 2, 3, 4, 5, 6, 7, targetVocab + 5 }, 0.001,
        UpdateRule::adam(1), 4), InvalidParameterError);

    EXPECT_NO_THROW(model.trainStepMultipleThread(source, decoderInput, expectedOutput, 0.001, UpdateRule::adam(1), 4));
}
