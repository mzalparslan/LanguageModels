#include "pch.h"
#include "BertPipeline.h"
#include <cmath>
#include <sstream>

namespace {
    using BertPipelineType = Pipeline<double, BertModel<double>, BertParameters<double>>;

    Logger& quietLogger() {
        static Logger logger(LogLevel::Critical);
        return logger;
    }

    // A small corpus of sentences in reading order.
    std::vector<std::string> corpus(std::size_t count) {
        const std::vector<std::string> words = { "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "home" };
        std::vector<std::string> sentences;
        for (std::size_t i = 0; i < count; i++) {
            std::string sentence;
            for (std::size_t j = 0; j < 4; j++) {
                sentence += words[(i * 3 + j) % words.size()] + " ";
            }
            sentences.push_back(sentence);
        }
        return sentences;
    }

    BertParameters<double> fastParameters(std::size_t epochs) {
        BertParameters<double> parameters;
        parameters.modelWidth = 16;
        parameters.layers = 1;
        parameters.maxLength = 16;
        parameters.epochs = epochs;
        parameters.logEveryEpochs = 0;
        return parameters;
    }
}

TEST(BertPipelineTest, TrainsAndEvaluatesBothObjectives) {
    BertPipelineType pipeline(fastParameters(3), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(corpus(30));

    BertMetrics metrics = pipeline.evaluate(corpus(20));

    EXPECT_EQ(metrics.examples, 20u);
    EXPECT_TRUE(std::isfinite(metrics.maskedLanguageModel.loss));
    EXPECT_GT(metrics.maskedLanguageModel.perplexity, 1.0);
    EXPECT_GE(metrics.maskedLanguageModel.accuracy, 0.0);
    EXPECT_LE(metrics.maskedLanguageModel.accuracy, 1.0);
    // Every example is scored for next-sentence prediction too.
    EXPECT_EQ(metrics.nextSentence.total(), 20u);
    EXPECT_GE(metrics.nextSentence.accuracy(), 0.0);
    EXPECT_LE(metrics.nextSentence.accuracy(), 1.0);
}

TEST(BertPipelineTest, EvaluationIsRepeatable) {
    BertPipelineType pipeline(fastParameters(2), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(corpus(30));

    BertMetrics first = pipeline.evaluate(corpus(20));
    BertMetrics second = pipeline.evaluate(corpus(20));

    EXPECT_EQ(first.maskedLanguageModel.loss, second.maskedLanguageModel.loss);
    EXPECT_EQ(first.nextSentence.accuracy(), second.nextSentence.accuracy());
}

TEST(BertPipelineTest, TrainingIsRepeatable) {
    BertPipelineType first(fastParameters(2), ExecutionStrategy::Sequential, quietLogger());
    BertPipelineType second(fastParameters(2), ExecutionStrategy::Sequential, quietLogger());

    first.train(corpus(30));
    second.train(corpus(30));

    EXPECT_EQ(first.modelAdapter().finalTrainingLoss(), second.modelAdapter().finalTrainingLoss());
}

TEST(BertPipelineTest, LongerTrainingLowersTheTrainingLoss) {
    BertPipelineType brief(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    BertPipelineType longer(fastParameters(8), ExecutionStrategy::Sequential, quietLogger());

    brief.train(corpus(30));
    longer.train(corpus(30));

    EXPECT_LT(longer.modelAdapter().finalTrainingLoss(), brief.modelAdapter().finalTrainingLoss());
}

TEST(BertPipelineTest, TheExecutionStrategyDoesNotChangeTheResult) {
    BertPipelineType sequential(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    BertPipelineType parallel(fastParameters(1), ExecutionStrategy::Parallel, quietLogger());

    sequential.train(corpus(20));
    parallel.train(corpus(20));

    EXPECT_EQ(sequential.evaluate(corpus(10)).maskedLanguageModel.loss,
        parallel.evaluate(corpus(10)).maskedLanguageModel.loss);
}

TEST(BertPipelineTest, VocabularyComesFromTheTrainingSentencesWithSpecialTokensFirst) {
    BertPipelineType pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train({ "alpha beta", "beta gamma" });

    const WordTokenizer& tokenizer = pipeline.modelAdapter().wordTokenizer();

    EXPECT_EQ(tokenizer.requireId("[PAD]"), 0u);
    EXPECT_EQ(tokenizer.requireId("[MASK]"), 4u);
    EXPECT_EQ(tokenizer.id("alpha"), 5u);
    EXPECT_EQ(tokenizer.size(), 5u + 3u);
}

TEST(BertPipelineTest, MaxVocabSizeTurnsRareWordsIntoUnknown) {
    BertParameters<double> parameters = fastParameters(1);
    parameters.maxVocabSize = 7; // 5 special tokens plus 2 words
    BertPipelineType pipeline(parameters, ExecutionStrategy::Sequential, quietLogger());

    pipeline.train({ "common common common", "common frequent frequent", "rare once" });

    const WordTokenizer& tokenizer = pipeline.modelAdapter().wordTokenizer();
    EXPECT_EQ(tokenizer.size(), 7u);
    EXPECT_EQ(tokenizer.id("rare"), tokenizer.requireId("[UNK]"));
}

TEST(BertPipelineTest, SentencesLongerThanTheMaximumLengthAreCutNotRejected) {
    BertParameters<double> parameters = fastParameters(1);
    parameters.maxLength = 9; // room for (9 - 3) / 2 = 3 words per sentence
    BertPipelineType pipeline(parameters, ExecutionStrategy::Sequential, quietLogger());
    const std::vector<std::string> longSentences = {
        "one two three four five six seven", "eight nine ten eleven twelve", "a b c d e f g h" };

    EXPECT_NO_THROW(pipeline.train(longSentences));
    EXPECT_NO_THROW((void)pipeline.evaluate(longSentences));
}

TEST(BertPipelineTest, SentencesWithoutWordsAreSkipped) {
    std::ostringstream output;
    Logger logger(LogLevel::Warning, output);
    BertPipelineType pipeline(fastParameters(1), ExecutionStrategy::Sequential, logger);
    std::vector<std::string> sentences = corpus(10);
    sentences.insert(sentences.begin() + 3, "   ");
    sentences.push_back("");

    pipeline.train(sentences);

    EXPECT_NE(output.str().find("2 training sentences skipped"), std::string::npos) << output.str();
    EXPECT_EQ(pipeline.evaluate(sentences).examples, 10u);
}

TEST(BertPipelineTest, RejectsInvalidSettingsAndData) {
    BertParameters<double> tooShort = fastParameters(1);
    tooShort.maxLength = 4;
    BertParameters<double> noEpochs = fastParameters(0);
    BertParameters<double> noMask = fastParameters(1);
    noMask.maskFraction = 0.0;
    BertParameters<double> overMask = fastParameters(1);
    overMask.maskFraction = 1.5;

    EXPECT_THROW(BertPipelineType(tooShort, ExecutionStrategy::Sequential, quietLogger()).train(corpus(10)),
        InvalidSizeError);
    EXPECT_THROW(BertPipelineType(noEpochs, ExecutionStrategy::Sequential, quietLogger()).train(corpus(10)),
        InvalidParameterSizeError);
    EXPECT_THROW(BertPipelineType(noMask, ExecutionStrategy::Sequential, quietLogger()).train(corpus(10)),
        InvalidParameterError);
    EXPECT_THROW(BertPipelineType(overMask, ExecutionStrategy::Sequential, quietLogger()).train(corpus(10)),
        InvalidParameterError);

    BertPipelineType pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    // Next-sentence prediction needs at least two sentences.
    EXPECT_THROW(pipeline.train({ "only one sentence" }), InvalidSizeError);
    EXPECT_FALSE(pipeline.isTrained());
}

TEST(BertPipelineTest, EvaluateNeedsATrainedModelAndAtLeastTwoSentences) {
    BertPipelineType pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());

    EXPECT_THROW((void)pipeline.evaluate(corpus(5)), PipelineStateError);
    EXPECT_THROW(pipeline.modelAdapter().trainedModel(), PipelineStateError);

    pipeline.train(corpus(10));
    EXPECT_THROW((void)pipeline.evaluate({ "just one" }), InvalidSizeError);
}

TEST(BertPipelineTest, EvaluateSkipsNothingWhenTestWordsAreUnknown) {
    BertPipelineType pipeline(fastParameters(1), ExecutionStrategy::Sequential, quietLogger());
    pipeline.train(corpus(10));

    // Words never seen in training are [UNK]: nothing can be masked in them, which
    // is an error rather than a silently empty score.
    EXPECT_THROW((void)pipeline.evaluate({ "zzz yyy", "xxx www" }), InvalidSizeError);
}

TEST(BertModelTest, PredictNextSentenceLogitsGivesTwoFiniteScores) {
    BertModel<double> model(20, 16, 1, 16);
    const std::vector<std::size_t> input = { 2, 5, 6, 3, 7, 3 }; // [CLS] a b [SEP] c [SEP]
    const std::vector<std::size_t> types = { 0, 0, 0, 0, 1, 1 };

    Tensor<double> first, second;
    model.predictNextSentenceLogits(input, types, first);
    model.predictNextSentenceLogits(input, types, second);

    EXPECT_EQ(first.shape, (std::vector<std::size_t>{ 1, 2 }));
    EXPECT_TRUE(std::isfinite(first[0]) && std::isfinite(first[1]));
    EXPECT_EQ(first.data, second.data) << "prediction does not change model";
}

TEST(BertModelTest, PredictNextSentenceLogitsMovesTowardsTheTrainedLabel) {
    BertModel<double> model(20, 16, 1, 16);
    const std::vector<std::size_t> input = { 2, 5, 6, 3, 7, 3 };
    const std::vector<std::size_t> types = { 0, 0, 0, 0, 1, 1 };
    const std::vector<std::size_t> mlmLabels = { 0, 0, 0, 0, 0, 0 };

    Tensor<double> before, after;
    model.predictNextSentenceLogits(input, types, before);
    for (std::size_t step = 1; step <= 40; step++) {
        // Label 1 = NotNext, with a masked position so MLM loss is defined.
        model.trainStep({ 2, 4, 6, 3, 7, 3 }, types, { 0, 5, 0, 0, 0, 0 }, 1, 0.01, UpdateRule::adam(step));
    }
    model.predictNextSentenceLogits(input, types, after);

    // margin of "NotNext" over "IsNext" grew.
    EXPECT_GT(after[1] - after[0], before[1] - before[0]);
}

TEST(BertModelTest, PredictNextSentenceLogitsValidatesItsInput) {
    BertModel<double> model(20, 16, 1, 16);
    Tensor<double> logits;

    EXPECT_THROW(model.predictNextSentenceLogits({}, {}, logits), InvalidSizeError);
    EXPECT_THROW(model.predictNextSentenceLogits({ 2, 3 }, { 0 }, logits), InvalidSizeError);
}
