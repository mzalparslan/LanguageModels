#include "BasicGPT.h"
#include "BertPipeline.h"
#include "DataLoader.h"
#include "DataSplitter.h"
#include "GptPipeline.h"
#include "Logger.h"
#include "MiniTransformerPipeline.h"
#include "VanillaRnnPipeline.h"

#include <iomanip>
#include <iostream>
#include <utility>

// Set by main(): --quick uses smaller datasets and fewer steps, --parallel
// trains Mini Transformer with several threads.
extern bool useQuickDataset;
extern ExecutionStrategy executionStrategy;

// Every stage below has same shape:
//
//   load data -> split it -> build a Pipeline<T, Model, Parameters>
//   -> pipeline.train(training part) -> pipeline.evaluate(test part)
//
// Only model, its Parameters and kind of data change.

namespace {
    Logger& logger() {
        static Logger stdoutLogger(LogLevel::Info, std::cout);
        return stdoutLogger;
    }

    void printLanguageModelResult(const LanguageModelMetrics& result) {
        std::cout << std::fixed << std::setprecision(3);
        auto row = [](const char* name, const Metrics& metrics) {
            std::cout << std::left << std::setw(26) << name << std::right
                << std::setw(8) << metrics.loss << std::setw(12) << metrics.perplexity
                << std::setw(10) << metrics.bpc << std::setw(9) << metrics.accuracy * 100 << "%\n";
        };
        std::cout << "\nHeld-out text (" << result.characters << " characters)\n"
            << std::left << std::setw(26) << "Model" << std::right << std::setw(8) << "loss"
            << std::setw(12) << "perplexity" << std::setw(10) << "bits/char" << std::setw(10) << "accuracy\n";
        row("Unigram baseline", result.unigram);
        row("Bigram baseline", result.bigram);
        row("Trained model", result.model);
        std::cout << "Beats unigram baseline: " << (result.model.perplexity < result.unigram.perplexity ? "yes" : "no")
            << ", bigram baseline: " << (result.model.perplexity < result.bigram.perplexity ? "yes" : "no") << "\n";
    }

    // first characters of Tiny Shakespeare in quick mode, all of it otherwise.
    std::vector<char> loadShakespeare() {
        std::vector<char> text = DataLoader::loadCharacters("tinyshakespeare.txt");
        if (useQuickDataset && text.size() > 200000) {
            text.resize(200000);
        }
        return text;
    }
}

/**
 * @brief Demo stage: English to French translation with MiniTransformer.
 *
 * Sentence pairs are shuffled and split 80/20, so held-out sentences come
 * from same file but are never trained on. pipeline reports BLEU, word
 * error rate and perplexity. With --parallel, every training step runs on
 * several threads.
 */
int testTranslationPipeline() {
    using TranslationPipeline = Pipeline<double, MiniTransformer<double>, TranslationParameters<double>>;

    auto samples = DataLoader::loadSentencePairs(useQuickDataset ? "fra_debug.txt" : "fra.txt", 5000);
    logger().info() << "Loaded " << samples.size() << " sentence pairs.";
    auto dataSet = DataSplitter::trainTestSplit(std::move(samples));

    TranslationParameters<double> parameters;
    // A full run over up to 5000 pairs takes a long time; --parallel makes it several times faster.
    parameters.epochs = useQuickDataset ? 100 : 30;
    parameters.logEveryEpochs = useQuickDataset ? 25 : 5;

    TranslationPipeline pipeline{ parameters, executionStrategy, logger() };
    pipeline.train(dataSet.trainingData);
    TranslationMetrics metrics = pipeline.evaluate(dataSet.testData);

    logger().info() << "Held-out sentences: " << metrics.sentenceCount
        << ", BLEU-4: " << metrics.bleu
        << ", word error rate: " << metrics.wordErrorRate
        << ", exact matches: " << metrics.exactMatchRate * 100 << "%"
        << ", perplexity: " << metrics.teacherForced.perplexity;

    // trained model translates any sentence.
    for (std::size_t i = 0; i < 3 && i < dataSet.testData.size(); i++) {
        const SentencePair& pair = dataSet.testData[i];
        std::cout << "Src: " << pair.source << " -> " << pipeline.modelAdapter().translate(pair.source)
            << "   (reference: " << pair.target << ")\n";
    }
    return 0;
}

/**
 * @brief Demo stage: a character-level language model with VanillaRNN on
 * Tiny Shakespeare, compared with unigram and bigram baselines.
 */
int testRnnPipeline() {
    using RnnPipeline = Pipeline<double, VanillaRNN<double>, RnnParameters<double>>;

    auto dataSet = DataSplitter::sequentialSplit(loadShakespeare());

    RnnParameters<double> parameters;
    parameters.steps = useQuickDataset ? 2000 : 50000;
    parameters.logEverySteps = parameters.steps / 5;

    RnnPipeline pipeline{ parameters, executionStrategy, logger() };
    pipeline.train(dataSet.trainingData);
    printLanguageModelResult(pipeline.evaluate(dataSet.testData));
    return 0;
}

/**
 * @brief Demo stage: a character-level language model with a decoder-only
 * GPT on Tiny Shakespeare, compared with unigram and bigram baselines.
 */
int testGptPipeline() {
    using GptPipelineType = Pipeline<double, BasicGPT, GptParameters<double>>;

    auto dataSet = DataSplitter::sequentialSplit(loadShakespeare());

    GptParameters<double> parameters;
    parameters.steps = useQuickDataset ? 1000 : 5000;
    parameters.logEverySteps = parameters.steps / 5;

    GptPipelineType pipeline{ parameters, executionStrategy, logger() };
    pipeline.train(dataSet.trainingData);
    printLanguageModelResult(pipeline.evaluate(dataSet.testData));
    return 0;
}

/**
 * @brief Demo stage: BERT on lines of Tiny Shakespeare, scored on hidden
 * words and on next-sentence prediction. Lines are split in reading order,
 * since "next sentence" needs order.
 */
int testBertPipeline() {
    using BertPipelineType = Pipeline<double, BertModel<double>, BertParameters<double>>;

    auto lines = DataLoader::loadLines("tinyshakespeare.txt", useQuickDataset ? 500 : 4000);
    logger().info() << "Loaded " << lines.size() << " lines.";
    auto dataSet = DataSplitter::sequentialSplit(std::move(lines));

    BertParameters<double> parameters;
    parameters.epochs = useQuickDataset ? 1 : 3;

    BertPipelineType pipeline{ parameters, executionStrategy, logger() };
    pipeline.train(dataSet.trainingData);
    BertMetrics metrics = pipeline.evaluate(dataSet.testData);

    logger().info() << "Held-out examples: " << metrics.examples
        << ", hidden-word perplexity: " << metrics.maskedLanguageModel.perplexity
        << ", hidden-word accuracy: " << metrics.maskedLanguageModel.accuracy * 100 << "%";
    logger().info() << "Next sentence: accuracy " << metrics.nextSentence.accuracy() * 100
        << "%, macro F1 " << metrics.nextSentence.macroF1();
    return 0;
}
