#include "pch.h"
#include "DataSplitter.h"
#include "VanillaRnnPipeline.h"
#include <cmath>
#include <sstream>

namespace {
    using RnnPipeline = Pipeline<double, VanillaRNN<double>, RnnParameters<double>>;

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

    RnnParameters<double> fastParameters(std::size_t steps) {
        RnnParameters<double> parameters;
        parameters.hiddenSize = 16;
        parameters.windowLength = 5;
        parameters.learningRate = 0.1;
        parameters.steps = steps;
        parameters.logEverySteps = 0;
        parameters.evaluationWindow = 20;
        return parameters;
    }
}

TEST(RnnPipelineTest, LearnsARepeatingTextAndBeatsBothBaselines) {
    auto dataset = DataSplitter::sequentialSplit(repeated("hello world ", 40));
    RnnPipeline pipeline(fastParameters(500), ExecutionStrategy::Sequential, quietLogger());

    pipeline.train(dataset.trainingData);
    LanguageModelMetrics metrics = pipeline.evaluate(dataset.testData);

    // text has 8 distinct characters, so a uniform guess would score 8.
    EXPECT_LT(metrics.model.perplexity, 2.0);
    EXPECT_LT(metrics.model.perplexity, metrics.unigram.perplexity);
    EXPECT_LT(metrics.model.perplexity, metrics.bigram.perplexity + 1e-9);
    EXPECT_GT(metrics.model.accuracy, 0.8);
}

TEST(RnnPipelineTest, ResultCountsTheScoredCharactersAndCarriesBaselines) {
    auto dataset = DataSplitter::sequentialSplit(repeated("abc", 100), 0.5);
    RnnPipeline pipeline(fastParameters(10), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(dataset.trainingData);

    LanguageModelMetrics metrics = pipeline.evaluate(dataset.testData);

    EXPECT_EQ(metrics.characters, dataset.testData.size() - 1);
    // Equal letter frequencies: unigram baseline is size of alphabet.
    EXPECT_NEAR(metrics.unigram.perplexity, 3.0, 0.05);
    EXPECT_LT(metrics.bigram.perplexity, 1.1);
    EXPECT_TRUE(std::isfinite(metrics.model.loss));
}

TEST(RnnPipelineTest, LongerTrainingLowersTheTrainingLoss) {
    auto dataset = DataSplitter::sequentialSplit(repeated("hello world ", 40));
    RnnParameters<double> brief = fastParameters(20);
    brief.logEverySteps = 10;
    RnnParameters<double> longer = fastParameters(400);
    longer.logEverySteps = 200;
    RnnPipeline first(brief, ExecutionStrategy::Sequential, quietLogger());
    RnnPipeline second(longer, ExecutionStrategy::Sequential, quietLogger());

    first.train(dataset.trainingData);
    second.train(dataset.trainingData);

    EXPECT_LT(second.modelAdapter().finalTrainingLoss(), first.modelAdapter().finalTrainingLoss());
}

TEST(RnnPipelineTest, TheExecutionStrategyDoesNotChangeTheResult) {
    auto dataset = DataSplitter::sequentialSplit(repeated("hello world ", 20));
    RnnPipeline sequential(fastParameters(50), ExecutionStrategy::Sequential, quietLogger());
    RnnPipeline parallel(fastParameters(50), ExecutionStrategy::Parallel, quietLogger());

    sequential.train(dataset.trainingData);
    parallel.train(dataset.trainingData);

    EXPECT_EQ(sequential.evaluate(dataset.testData).model.loss, parallel.evaluate(dataset.testData).model.loss);
}

TEST(RnnPipelineTest, CharactersMissingFromTheTrainingTextAreScoredNotRejected) {
    RnnPipeline pipeline(fastParameters(20), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(repeated("abab", 30));

    // 'z' never occurred in training: it becomes unknown character.
    LanguageModelMetrics metrics = pipeline.evaluate(repeated("abzab", 5));

    EXPECT_TRUE(std::isfinite(metrics.model.loss));
    EXPECT_GT(metrics.model.perplexity, 1.0);
}

TEST(RnnPipelineTest, TrainingWrapsAroundWhenItRunsOutOfText) {
    // 40 characters, windows of 5: 8 windows only, but 100 steps are requested.
    RnnPipeline pipeline(fastParameters(100), ExecutionStrategy::Sequential, quietLogger());

    EXPECT_NO_THROW(pipeline.train(repeated("abcd", 10)));
}

TEST(RnnPipelineTest, ProgressIsLoggedEveryConfiguredNumberOfSteps) {
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);
    RnnParameters<double> parameters = fastParameters(30);
    parameters.logEverySteps = 10;
    RnnPipeline pipeline(parameters, ExecutionStrategy::Sequential, logger);

    pipeline.train(repeated("abc", 50));

    EXPECT_NE(output.str().find("Step 10 average loss per character"), std::string::npos) << output.str();
    EXPECT_NE(output.str().find("Step 20 average loss per character"), std::string::npos);
    EXPECT_NE(output.str().find("Step 30 average loss per character"), std::string::npos);
    EXPECT_EQ(output.str().find("Step 5 average"), std::string::npos);
}

TEST(RnnPipelineTest, RejectsInvalidSettingsAndData) {
    RnnParameters<double> noHidden = fastParameters(1);
    noHidden.hiddenSize = 0;
    RnnParameters<double> noWindow = fastParameters(1);
    noWindow.windowLength = 0;
    RnnParameters<double> noEvaluationWindow = fastParameters(1);
    noEvaluationWindow.evaluationWindow = 0;

    EXPECT_THROW(RnnPipeline(noHidden, ExecutionStrategy::Sequential, quietLogger()).train(repeated("abc", 10)),
        InvalidParameterSizeError);
    EXPECT_THROW(RnnPipeline(noWindow, ExecutionStrategy::Sequential, quietLogger()).train(repeated("abc", 10)),
        InvalidParameterSizeError);
    EXPECT_THROW(RnnPipeline(noEvaluationWindow, ExecutionStrategy::Sequential, quietLogger()).train(repeated("abc", 10)),
        InvalidParameterSizeError);

    RnnPipeline pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    // A window of 5 needs at least 6 characters.
    EXPECT_THROW(pipeline.train(repeated("abc", 1)), InvalidSizeError);
    EXPECT_FALSE(pipeline.isTrained());
}

TEST(RnnPipelineTest, EvaluateNeedsATrainedModelAndAtLeastTwoCharacters) {
    RnnPipeline pipeline(fastParameters(5), ExecutionStrategy::Sequential, quietLogger());

    EXPECT_THROW((void)pipeline.evaluate(repeated("abc", 5)), PipelineStateError);
    EXPECT_THROW(pipeline.modelAdapter().trainedModel(), PipelineStateError);

    pipeline.train(repeated("abc", 20));
    EXPECT_THROW((void)pipeline.evaluate(std::vector<char>{ 'a' }), InvalidSizeError);
    EXPECT_NO_THROW((void)pipeline.evaluate(std::vector<char>{ 'a', 'b' }));
}
