#include "pch.h"
#include "BasicGPT.h"
#include "DataSplitter.h"
#include "GptPipeline.h"
#include <cmath>
#include <sstream>

namespace {
    using GptPipelineType = Pipeline<double, BasicGPT, GptParameters<double>>;

    Logger& quietLogger() {
        static Logger logger(LogLevel::Critical);
        return logger;
    }

    std::vector<char> repeated(const std::string& unit, std::size_t times) {
        std::vector<char> text;
        for (std::size_t i = 0; i < times; i++) {
            text.insert(text.end(), unit.begin(), unit.end());
        }
        return text;
    }

    GptParameters<double> fastParameters(std::size_t steps) {
        GptParameters<double> parameters;
        parameters.layers = 1;
        parameters.steps = steps;
        parameters.logEverySteps = 0;
        return parameters;
    }
}

TEST(GptPipelineTest, LearnsARepeatingTextAndBeatsTheBaselines) {
    auto dataset = DataSplitter::sequentialSplit(repeated("abc", 200));
    GptPipelineType pipeline(fastParameters(300), ExecutionStrategy::Sequential, quietLogger());

    pipeline.train(dataset.trainingData);
    LanguageModelMetrics metrics = pipeline.evaluate(dataset.testData);

    // Three equally frequent letters: unigram baseline is about 3, and a
    // model that has learned pattern does much better.
    EXPECT_NEAR(metrics.unigram.perplexity, 3.0, 0.1);
    EXPECT_LT(metrics.model.perplexity, metrics.unigram.perplexity);
    EXPECT_GT(metrics.model.accuracy, 0.5);
}

TEST(GptPipelineTest, LongerTrainingLowersTheTrainingLoss) {
    auto text = repeated("abc", 200);
    GptParameters<double> brief = fastParameters(20);
    brief.logEverySteps = 10;
    GptParameters<double> longer = fastParameters(200);
    longer.logEverySteps = 100;
    GptPipelineType first(brief, ExecutionStrategy::Sequential, quietLogger());
    GptPipelineType second(longer, ExecutionStrategy::Sequential, quietLogger());

    first.train(text);
    second.train(text);

    EXPECT_LT(second.modelAdapter().finalTrainingLoss(), first.modelAdapter().finalTrainingLoss());
}

TEST(GptPipelineTest, ResultCountsTheScoredCharacters) {
    auto dataset = DataSplitter::sequentialSplit(repeated("abc", 100), 0.5);
    GptPipelineType pipeline(fastParameters(5), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(dataset.trainingData);

    LanguageModelMetrics metrics = pipeline.evaluate(dataset.testData);

    EXPECT_EQ(metrics.characters, dataset.testData.size() - 1);
    EXPECT_TRUE(std::isfinite(metrics.model.loss));
}

TEST(GptPipelineTest, TheContextDefaultsToTheModelsMaximumAndCanBeShortened) {
    GptParameters<double> shorter = fastParameters(5);
    shorter.contextLength = 8;
    GptPipelineType defaulted(fastParameters(5), ExecutionStrategy::Sequential, quietLogger());
    GptPipelineType custom(shorter, ExecutionStrategy::Sequential, quietLogger());

    defaulted.train(repeated("abc", 50));
    custom.train(repeated("abc", 50));

    EXPECT_EQ(defaulted.modelAdapter().trainedModel().maxLen, BasicGPTConfig::maxSeqLen);
    EXPECT_EQ(custom.modelAdapter().trainedModel().maxLen, 8u);
}

TEST(GptPipelineTest, TheExecutionStrategyDoesNotChangeTheResult) {
    auto dataset = DataSplitter::sequentialSplit(repeated("abc", 100));
    GptPipelineType sequential(fastParameters(20), ExecutionStrategy::Sequential, quietLogger());
    GptPipelineType parallel(fastParameters(20), ExecutionStrategy::Parallel, quietLogger());

    sequential.train(dataset.trainingData);
    parallel.train(dataset.trainingData);

    EXPECT_EQ(sequential.evaluate(dataset.testData).model.loss, parallel.evaluate(dataset.testData).model.loss);
}

TEST(GptPipelineTest, TheTrainedModelCanGenerateText) {
    GptPipelineType pipeline(fastParameters(300), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(repeated("abc", 200));

    // 'a' is id 1 (id 0 is unknown character), then 'b' = 2, 'c' = 3.
    auto generated = pipeline.modelAdapter().trainedModel().generate({ 1, 2 }, 4);

    EXPECT_EQ(generated.size(), 6u);
    EXPECT_EQ(generated[0], 1u);
    EXPECT_EQ(generated[1], 2u);
}

TEST(GptPipelineTest, ProgressIsLoggedEveryConfiguredNumberOfSteps) {
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);
    GptParameters<double> parameters = fastParameters(20);
    parameters.logEverySteps = 10;
    GptPipelineType pipeline(parameters, ExecutionStrategy::Sequential, logger);

    pipeline.train(repeated("abc", 100));

    EXPECT_NE(output.str().find("Step 10 average loss per character"), std::string::npos) << output.str();
    EXPECT_NE(output.str().find("Step 20 average loss per character"), std::string::npos);
}

TEST(GptPipelineTest, RejectsInvalidSettingsAndData) {
    GptParameters<double> noLayers = fastParameters(1);
    noLayers.layers = 0;
    GptParameters<double> tooLong = fastParameters(1);
    tooLong.contextLength = BasicGPTConfig::maxSeqLen + 1;

    EXPECT_THROW(GptPipelineType(noLayers, ExecutionStrategy::Sequential, quietLogger()).train(repeated("abc", 50)),
        InvalidParameterSizeError);
    EXPECT_THROW(GptPipelineType(tooLong, ExecutionStrategy::Sequential, quietLogger()).train(repeated("abc", 50)),
        InvalidSizeError);

    GptPipelineType pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    // A context of 20 needs at least 21 characters.
    EXPECT_THROW(pipeline.train(repeated("abc", 5)), InvalidSizeError);
    EXPECT_FALSE(pipeline.isTrained());
}

TEST(GptPipelineTest, EvaluateNeedsATrainedModelAndAtLeastTwoCharacters) {
    GptPipelineType pipeline(fastParameters(5), ExecutionStrategy::Sequential, quietLogger());

    EXPECT_THROW((void)pipeline.evaluate(repeated("abc", 5)), PipelineStateError);
    EXPECT_THROW(pipeline.modelAdapter().trainedModel(), PipelineStateError);

    pipeline.train(repeated("abc", 50));
    EXPECT_THROW((void)pipeline.evaluate(std::vector<char>{ 'a' }), InvalidSizeError);
}
