#include "pch.h"
#include "GpuVocabularyHead.h"
#include "MiniTransformerPipeline.h"
#include "VanillaRnnPipeline.h"
#include <cmath>
#include <iostream>
#include <sstream>

#define SKIP_WITHOUT_GPU() \
    if (!gpu::isAvailable<double>()) { \
        std::cout << "[  SKIPPED ] no GPU that computes in double (CUDA), or this build has none\n"; \
        return; \
    }

namespace {
    using TranslationPipeline = Pipeline<double, MiniTransformer<double>, TranslationParameters<double>>;

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

    TranslationParameters<double> fastParameters(std::size_t epochs) {
        TranslationParameters<double> parameters;
        parameters.epochs = epochs;
        parameters.learningRate = 0.005;
        parameters.logEveryEpochs = 0;
        return parameters;
    }

    // Many distinct words, so the vocabulary-sized work is what dominates.
    std::vector<SentencePair> wideVocabularyPairs(std::size_t count) {
        std::vector<SentencePair> pairs;
        for (std::size_t i = 0; i < count; i++) {
            std::string word;
            for (std::size_t n = i + 1; n > 0; n /= 26) {
                word += static_cast<char>('a' + (n % 26));
            }
            pairs.push_back({ "source " + word + " here", "target " + word });
        }
        return pairs;
    }
}

TEST(TranslationPipelineDeviceTest, DeviceIsAnExecutionStrategyWithAName) {
    EXPECT_STREQ(toString(ExecutionStrategy::Device), "Device");
}

TEST(TranslationPipelineDeviceTest, TrainsAModelThatTranslatesItsTrainingSentences) {
    SKIP_WITHOUT_GPU();
    TranslationPipeline pipeline(fastParameters(300), ExecutionStrategy::Device, quietLogger());

    pipeline.train(toyPairs());

    // The weights were trained on the GPU and copied back, so the CPU model translates.
    EXPECT_EQ(pipeline.modelAdapter().translate("hello there"), "bonjour la");
    EXPECT_EQ(pipeline.modelAdapter().translate("thank you"), "merci vous");
    EXPECT_DOUBLE_EQ(pipeline.evaluate(toyPairs()).exactMatchRate, 1.0);
}

TEST(TranslationPipelineDeviceTest, ARunOfAFewHundredStepsMatchesSequentialToWithinRounding) {
    SKIP_WITHOUT_GPU();
    // One epoch is 100 Adam steps: the two runs differ only by rounding (about 1e-11).
    const auto pairs = wideVocabularyPairs(100);
    const std::vector<SentencePair> someTestPairs(pairs.begin(), pairs.begin() + 10);
    TranslationPipeline sequential(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    TranslationPipeline onDevice(fastParameters(1), ExecutionStrategy::Device, quietLogger());

    sequential.train(pairs);
    onDevice.train(pairs);
    const double sequentialLoss = sequential.modelAdapter().finalTrainingLoss();
    const double gpuLoss = onDevice.modelAdapter().finalTrainingLoss();
    const TranslationMetrics a = sequential.evaluate(someTestPairs);
    const TranslationMetrics b = onDevice.evaluate(someTestPairs);

    EXPECT_NEAR(gpuLoss, sequentialLoss, 1e-6 * sequentialLoss);
    EXPECT_NEAR(b.teacherForced.loss, a.teacherForced.loss, 1e-6 * a.teacherForced.loss);
}

TEST(TranslationPipelineDeviceTest, LongerRunsTrainAsWellEvenThoughTheyDriftApartFromSequential) {
    SKIP_WITHOUT_GPU();
    // Training is chaotic: two runs that differ by a rounding error drift apart over a
    // few hundred steps, as any two runs that are not bit-identical do (a different
    // compiler or thread order would do the same). What must hold is that the GPU run
    // learns just as well, which is why this compares the quality reached, not the values.
    const auto pairs = wideVocabularyPairs(100);
    TranslationPipeline brief(fastParameters(1), ExecutionStrategy::Device, quietLogger());
    TranslationPipeline longer(fastParameters(4), ExecutionStrategy::Device, quietLogger());
    TranslationPipeline sequential(fastParameters(4), ExecutionStrategy::Sequential, quietLogger());

    brief.train(pairs);
    longer.train(pairs);
    sequential.train(pairs);

    const double briefLoss = brief.modelAdapter().finalTrainingLoss();
    const double gpuLoss = longer.modelAdapter().finalTrainingLoss();
    const double cpuLoss = sequential.modelAdapter().finalTrainingLoss();
    EXPECT_LT(gpuLoss, 0.5 * briefLoss);
    EXPECT_LT(cpuLoss, 0.5 * briefLoss);
    EXPECT_LT(gpuLoss, 2.0 * cpuLoss);
    EXPECT_LT(cpuLoss, 2.0 * gpuLoss);
}

TEST(TranslationPipelineDeviceTest, TheGpuIsNamedInTheLog) {
    SKIP_WITHOUT_GPU();
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);
    TranslationPipeline pipeline(fastParameters(1), ExecutionStrategy::Device, logger);

    pipeline.train(toyPairs());

    EXPECT_NE(output.str().find("execution: Device"), std::string::npos) << output.str();
    EXPECT_NE(output.str().find("GPU: "), std::string::npos);
}

TEST(TranslationPipelineDeviceTest, WithoutAGpuTrainingOnTheDeviceThrowsDeviceError) {
    if (gpu::isAvailable<double>()) {
        return; // only meaningful where there is nothing to use
    }
    TranslationPipeline pipeline(fastParameters(1), ExecutionStrategy::Device, quietLogger());

    EXPECT_THROW(pipeline.train(toyPairs()), DeviceError);

    EXPECT_FALSE(pipeline.isTrained());
}

TEST(TranslationPipelineDeviceTest, OtherModelsAcceptTheStrategyAndTrainTheSameWay) {
    using RnnPipeline = Pipeline<double, VanillaRNN<double>, RnnParameters<double>>;
    RnnParameters<double> parameters;
    parameters.hiddenSize = 16;
    parameters.windowLength = 5;
    parameters.learningRate = 0.1;
    parameters.steps = 30;
    parameters.logEverySteps = 0;
    std::vector<char> text;
    for (int i = 0; i < 40; i++) {
        text.insert(text.end(), { 'a', 'b', 'c', ' ' });
    }
    RnnPipeline sequential(parameters, ExecutionStrategy::Sequential, quietLogger());
    RnnPipeline deviceStrategy(parameters, ExecutionStrategy::Device, quietLogger());

    sequential.train(text);
    deviceStrategy.train(text);

    // The RNN ignores the strategy, so this needs no GPU and the results are identical.
    EXPECT_EQ(sequential.evaluate(text).model.loss, deviceStrategy.evaluate(text).model.loss);
}
