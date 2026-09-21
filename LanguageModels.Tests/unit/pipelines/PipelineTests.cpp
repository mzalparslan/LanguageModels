#include "pch.h"
#include "Pipeline.h"
#include <sstream>
#include <stdexcept>

// generic Pipeline is tested with a made-up model, so what is checked is
// pipeline's own behaviour (ordering, validation, what it hands to the
// adapter), not any real model.

namespace {
    class FakeModel {};

    class FakeParameters {
    public:
        bool failTraining = false;
    };

    class FakeResult {
    public:
        std::size_t sampleCount = 0;
    };
}

// adapter a model provides to be usable in a Pipeline.
template <>
class ModelAdapter<double, FakeModel, FakeParameters> {
public:
    using Sample = int;
    using Parameters = FakeParameters;
    using Result = FakeResult;

    explicit ModelAdapter(Parameters parameters) : parameters(parameters) {}

    void train(const std::vector<int>& samples, ExecutionStrategy strategy, Logger&) {
        trainCalls++;
        lastStrategy = strategy;
        lastTrainingSize = samples.size();
        if (parameters.failTraining) {
            throw std::runtime_error("training failed");
        }
    }

    FakeResult evaluate(const std::vector<int>& samples, Logger&) {
        FakeResult result;
        result.sampleCount = samples.size();
        return result;
    }

    Parameters parameters;
    int trainCalls = 0;
    ExecutionStrategy lastStrategy = ExecutionStrategy::Sequential;
    std::size_t lastTrainingSize = 0;
};

namespace {
    using FakePipeline = Pipeline<double, FakeModel, FakeParameters>;
}

TEST(PipelineTest, TheAdapterConceptAcceptsAdaptersAndRejectsOtherTypes) {
    static_assert(PipelineAdapter<ModelAdapter<double, FakeModel, FakeParameters>>);
    static_assert(!PipelineAdapter<int>);
    static_assert(!PipelineAdapter<FakeModel>);
    SUCCEED();
}

TEST(PipelineTest, PipelineExposesTheAdapterTypes) {
    static_assert(std::is_same_v<FakePipeline::Sample, int>);
    static_assert(std::is_same_v<FakePipeline::Result, FakeResult>);
    SUCCEED();
}

TEST(PipelineTest, TrainThenEvaluateHandsTheDataToTheAdapter) {
    std::ostringstream logOutput;
    Logger logger(LogLevel::Info, logOutput);
    FakePipeline pipeline(FakeParameters{}, ExecutionStrategy::Sequential, logger);

    pipeline.train({ 1, 2, 3, 4 });
    FakeResult result = pipeline.evaluate({ 5, 6 });

    EXPECT_EQ(pipeline.modelAdapter().lastTrainingSize, 4u);
    EXPECT_EQ(result.sampleCount, 2u);
}

TEST(PipelineTest, IsNotTrainedUntilTrainCompletes) {
    Logger logger(LogLevel::Critical);
    FakePipeline pipeline(FakeParameters{}, ExecutionStrategy::Sequential, logger);

    EXPECT_FALSE(pipeline.isTrained());
    pipeline.train({ 1 });
    EXPECT_TRUE(pipeline.isTrained());
}

TEST(PipelineTest, EvaluateBeforeTrainThrowsPipelineStateError) {
    Logger logger(LogLevel::Critical);
    FakePipeline pipeline(FakeParameters{}, ExecutionStrategy::Sequential, logger);

    EXPECT_THROW((void)pipeline.evaluate({ 1 }), PipelineStateError);
}

TEST(PipelineTest, PipelineStateErrorIsALogicError) {
    static_assert(std::is_base_of_v<std::logic_error, PipelineStateError>);
    SUCCEED();
}

TEST(PipelineTest, EmptyDataIsRejected) {
    Logger logger(LogLevel::Critical);
    FakePipeline pipeline(FakeParameters{}, ExecutionStrategy::Sequential, logger);

    EXPECT_THROW(pipeline.train({}), InvalidSizeError);
    EXPECT_EQ(pipeline.modelAdapter().trainCalls, 0) << "the adapter is not asked to train on nothing";

    pipeline.train({ 1 });
    EXPECT_THROW((void)pipeline.evaluate({}), InvalidSizeError);
}

TEST(PipelineTest, TheExecutionStrategyReachesTheAdapter) {
    Logger logger(LogLevel::Critical);
    FakePipeline sequential(FakeParameters{}, ExecutionStrategy::Sequential, logger);
    FakePipeline parallel(FakeParameters{}, ExecutionStrategy::Parallel, logger);

    sequential.train({ 1 });
    parallel.train({ 1 });

    EXPECT_EQ(sequential.modelAdapter().lastStrategy, ExecutionStrategy::Sequential);
    EXPECT_EQ(parallel.modelAdapter().lastStrategy, ExecutionStrategy::Parallel);
    EXPECT_EQ(parallel.executionStrategy(), ExecutionStrategy::Parallel);
}

TEST(PipelineTest, SequentialIsTheDefaultStrategy) {
    FakePipeline defaulted{ FakeParameters{} };

    EXPECT_EQ(defaulted.executionStrategy(), ExecutionStrategy::Sequential);
}

TEST(PipelineTest, AFailedTrainingLeavesThePipelineUntrained) {
    Logger logger(LogLevel::Critical);
    FakeParameters failing;
    failing.failTraining = true;
    FakePipeline pipeline(failing, ExecutionStrategy::Sequential, logger);

    EXPECT_THROW(pipeline.train({ 1 }), std::runtime_error);

    EXPECT_FALSE(pipeline.isTrained());
    EXPECT_THROW((void)pipeline.evaluate({ 1 }), PipelineStateError);
}

TEST(PipelineTest, AFailedRetrainingInvalidatesEarlierTraining) {
    Logger logger(LogLevel::Critical);
    FakePipeline pipeline(FakeParameters{}, ExecutionStrategy::Sequential, logger);
    pipeline.train({ 1 });
    ASSERT_TRUE(pipeline.isTrained());

    pipeline.modelAdapter().parameters.failTraining = true;
    EXPECT_THROW(pipeline.train({ 1 }), std::runtime_error);

    EXPECT_FALSE(pipeline.isTrained());
}

TEST(PipelineTest, ProgressIsLoggedAndTheStrategyIsNamed) {
    std::ostringstream logOutput;
    Logger logger(LogLevel::Info, logOutput);
    FakePipeline pipeline(FakeParameters{}, ExecutionStrategy::Parallel, logger);

    pipeline.train({ 1, 2, 3 });

    EXPECT_NE(logOutput.str().find("Training started. Samples: 3"), std::string::npos) << logOutput.str();
    EXPECT_NE(logOutput.str().find("Parallel"), std::string::npos);
    EXPECT_NE(logOutput.str().find("Training completed."), std::string::npos);
}

TEST(ExecutionStrategyTest, HasNamesForLogging) {
    EXPECT_STREQ(toString(ExecutionStrategy::Sequential), "Sequential");
    EXPECT_STREQ(toString(ExecutionStrategy::Parallel), "Parallel");
}
