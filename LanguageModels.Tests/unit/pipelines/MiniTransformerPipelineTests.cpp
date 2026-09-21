#include "pch.h"
#include "MiniTransformerPipeline.h"
#include "TestSupport.h"

namespace {
    using TranslationPipeline = Pipeline<double, MiniTransformer<double>, TranslationParameters<double>>;

    Logger& quietLogger() {
        static Logger logger(LogLevel::Critical);
        return logger;
    }

    // A tiny fixed "language": each source sentence has one obvious translation.
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

    // Many distinct words, so tables are big enough for parallel step to
    // split their update between threads (the matrix multiplies need a still larger
    // vocabulary; MiniTransformerThreadingTests covers those).
    std::vector<SentencePair> wideVocabularyPairs(std::size_t count) {
        std::vector<SentencePair> pairs;
        for (std::size_t i = 0; i < count; i++) {
            // Words are built from letters only, because data is normalized text.
            std::string word;
            for (std::size_t n = i + 1; n > 0; n /= 26) {
                word += static_cast<char>('a' + (n % 26));
            }
            pairs.push_back({ "source " + word + " here", "target " + word });
        }
        return pairs;
    }
}

TEST(TranslationPipelineTest, TrainsAModelThatTranslatesAndEvaluateReportsItsQuality) {
    TranslationPipeline pipeline(fastParameters(300), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(toyPairs());

    // model translates its training sentences...
    EXPECT_EQ(pipeline.modelAdapter().translate("hello there"), "bonjour la");
    EXPECT_EQ(pipeline.modelAdapter().translate("thank you"), "merci vous");

    // ...and evaluate() reports how well it does on them.
    TranslationMetrics metrics = pipeline.evaluate(toyPairs());

    EXPECT_EQ(metrics.sentenceCount, 4u);
    EXPECT_DOUBLE_EQ(metrics.exactMatchRate, 1.0);
    EXPECT_DOUBLE_EQ(metrics.wordErrorRate, 0.0);
    // Two-word sentences have no 3-gram or 4-gram, so plain BLEU-4 cannot be positive.
    EXPECT_GE(metrics.bleu, 0.0);
    EXPECT_LE(metrics.bleu, 1.0);
    // A trained model is confident in right words.
    EXPECT_LT(metrics.teacherForced.perplexity, 1.5);
    EXPECT_GT(metrics.teacherForced.accuracy, 0.9);
}

TEST(TranslationPipelineTest, LongerTrainingLowersTheTrainingLoss) {
    TranslationPipeline brief(fastParameters(2), ExecutionStrategy::Sequential, quietLogger());
    TranslationPipeline longer(fastParameters(60), ExecutionStrategy::Sequential, quietLogger());

    brief.train(toyPairs());
    longer.train(toyPairs());

    EXPECT_LT(longer.modelAdapter().finalTrainingLoss(), brief.modelAdapter().finalTrainingLoss());
}

TEST(TranslationPipelineTest, ParallelExecutionGivesTheSameResultsAsSequential) {
    const auto pairs = wideVocabularyPairs(320);
    const std::vector<SentencePair> someTestPairs(pairs.begin(), pairs.begin() + 10);
    TranslationParameters<double> parameters = fastParameters(1);
    parameters.threadCount = 4;
    TranslationPipeline sequential(parameters, ExecutionStrategy::Sequential, quietLogger());
    TranslationPipeline parallel(parameters, ExecutionStrategy::Parallel, quietLogger());

    sequential.train(pairs);
    parallel.train(pairs);
    TranslationMetrics a = sequential.evaluate(someTestPairs);
    TranslationMetrics b = parallel.evaluate(someTestPairs);

    // Bit for bit, not just close.
    EXPECT_EQ(sequential.modelAdapter().finalTrainingLoss(), parallel.modelAdapter().finalTrainingLoss());
    EXPECT_EQ(a.teacherForced.loss, b.teacherForced.loss);
    EXPECT_EQ(a.teacherForced.accuracy, b.teacherForced.accuracy);
    EXPECT_EQ(a.bleu, b.bleu);
    EXPECT_EQ(a.wordErrorRate, b.wordErrorRate);
    EXPECT_EQ(a.exactMatchRate, b.exactMatchRate);

    Tensor<double> logitsA, logitsB;
    sequential.modelAdapter().trainedModel().forward({ 3, 4, 5 }, { 1, 3, 4 }, logitsA);
    parallel.modelAdapter().trainedModel().forward({ 3, 4, 5 }, { 1, 3, 4 }, logitsB);
    EXPECT_TRUE(testsupport::tensorsEqual(logitsA, logitsB));
}

TEST(TranslationPipelineTest, TrainingIsRepeatable) {
    TranslationPipeline first(fastParameters(5), ExecutionStrategy::Sequential, quietLogger());
    TranslationPipeline second(fastParameters(5), ExecutionStrategy::Sequential, quietLogger());

    first.train(toyPairs());
    second.train(toyPairs());

    EXPECT_EQ(first.modelAdapter().finalTrainingLoss(), second.modelAdapter().finalTrainingLoss());
    EXPECT_EQ(first.evaluate(toyPairs()).teacherForced.loss, second.evaluate(toyPairs()).teacherForced.loss);
}

TEST(TranslationPipelineTest, HeldOutSentencesWithNewWordsAreScoredWithoutErrors) {
    TranslationPipeline pipeline(fastParameters(20), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(toyPairs());

    // Neither side of this pair was ever seen: its words become <UNK>.
    TranslationMetrics metrics = pipeline.evaluate({ { "completely new words", "mots tout neufs" } });

    EXPECT_EQ(metrics.sentenceCount, 1u);
    EXPECT_TRUE(std::isfinite(metrics.teacherForced.loss));
    EXPECT_DOUBLE_EQ(metrics.exactMatchRate, 0.0);
    EXPECT_GT(metrics.wordErrorRate, 0.0);
}

TEST(TranslationPipelineTest, VocabulariesAreBuiltFromTheTrainingPairsOnly) {
    TranslationPipeline pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(toyPairs());

    // Target: <UNK> <SOS> <EOS> + 7 distinct words (bonjour la bon matin merci vous voir).
    Tensor<double> logits;
    pipeline.modelAdapter().trainedModel().forward({ 1 }, { 1 }, logits);
    EXPECT_EQ(logits.shape[1], 3u + 7u);
}

TEST(TranslationPipelineTest, UnusablePairsAreSkipped) {
    std::vector<SentencePair> pairs = toyPairs();
    pairs.push_back({ "", "empty source" });
    pairs.push_back({ "empty target", "" });
    std::string tooLong;
    for (std::size_t i = 0; i < MiniTransformerConfig::maxSeqLen + 1; i++) {
        tooLong += "word ";
    }
    pairs.push_back({ tooLong, "short" });
    TranslationPipeline pipeline(fastParameters(2), ExecutionStrategy::Sequential, quietLogger());

    EXPECT_NO_THROW(pipeline.train(pairs));
    // bad pairs are skipped in evaluation too, so only four good ones count.
    EXPECT_EQ(pipeline.evaluate(pairs).sentenceCount, 4u);
}

TEST(TranslationPipelineTest, SkippedPairsAreReportedAsAWarning) {
    std::ostringstream output;
    Logger logger(LogLevel::Warning, output);
    std::vector<SentencePair> pairs = toyPairs();
    pairs.push_back({ "", "nothing" });
    TranslationPipeline pipeline(fastParameters(1), ExecutionStrategy::Sequential, logger);

    pipeline.train(pairs);

    EXPECT_NE(output.str().find("[WARNING] 1 training pairs skipped"), std::string::npos) << output.str();
}

TEST(TranslationPipelineTest, RejectsDataWithNoUsablePair) {
    TranslationPipeline pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());

    EXPECT_THROW(pipeline.train({ { "", "x" }, { "y", "" } }), InvalidSizeError);
    EXPECT_FALSE(pipeline.isTrained());

    pipeline.train(toyPairs());
    EXPECT_THROW((void)pipeline.evaluate({ { "", "x" } }), InvalidSizeError);
}

TEST(TranslationPipelineTest, EvaluateAndTranslateBeforeTrainingThrowPipelineStateError) {
    TranslationPipeline pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());

    EXPECT_THROW((void)pipeline.evaluate(toyPairs()), PipelineStateError);
    EXPECT_THROW(pipeline.modelAdapter().translate("hello"), PipelineStateError);
    EXPECT_THROW(pipeline.modelAdapter().trainedModel(), PipelineStateError);
}

TEST(TranslationPipelineTest, TranslateValidatesTheSentence) {
    TranslationPipeline pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(toyPairs());

    EXPECT_THROW(pipeline.modelAdapter().translate(""), InvalidSizeError);
    EXPECT_THROW(pipeline.modelAdapter().translate("123 !!!"), InvalidSizeError);
    // Text is normalized like training data.
    EXPECT_NO_THROW(pipeline.modelAdapter().translate("HELLO, There!"));
}

TEST(TranslationPipelineTest, TrainingAgainStartsFromANewModel) {
    TranslationPipeline pipeline(fastParameters(5), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(toyPairs());
    const double firstLoss = pipeline.modelAdapter().finalTrainingLoss();

    pipeline.train(toyPairs());

    // Same data, same seed: an identical run, not five more epochs on old model.
    EXPECT_EQ(pipeline.modelAdapter().finalTrainingLoss(), firstLoss);
}

TEST(TranslationPipelineTest, ProgressIsLoggedEveryConfiguredNumberOfEpochs) {
    std::ostringstream output;
    Logger logger(LogLevel::Info, output);
    TranslationParameters<double> parameters = fastParameters(4);
    parameters.logEveryEpochs = 2;
    TranslationPipeline pipeline(parameters, ExecutionStrategy::Sequential, logger);

    pipeline.train(toyPairs());

    EXPECT_NE(output.str().find("Epoch 0 average loss"), std::string::npos) << output.str();
    EXPECT_EQ(output.str().find("Epoch 1 average loss"), std::string::npos);
    EXPECT_NE(output.str().find("Epoch 2 average loss"), std::string::npos);
}
