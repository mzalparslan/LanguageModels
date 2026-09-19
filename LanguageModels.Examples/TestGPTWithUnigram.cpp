#include <iostream>
#include <vector>
#include <string>
#include "BasicGPTWithMoE.h"
#include "UnigramTokenizer.h"

namespace {

/**
 * @brief Converts tokenizer ids (int) to model token ids (size_t).
 * Tokenizer ids are vocabulary indices, so they are never negative.
 */
std::vector<std::size_t> toModelIds(const std::vector<int>& tokenizerIds) {
    std::vector<std::size_t> modelIds;
    modelIds.reserve(tokenizerIds.size());
    for (int id : tokenizerIds) {
        modelIds.push_back(static_cast<std::size_t>(id));
    }
    return modelIds;
}

/**
 * @brief Converts model token ids (size_t) back to tokenizer ids (int).
 */
std::vector<int> toTokenizerIds(const std::vector<std::size_t>& modelIds) {
    std::vector<int> tokenizerIds;
    tokenizerIds.reserve(modelIds.size());
    for (std::size_t id : modelIds) {
        tokenizerIds.push_back(static_cast<int>(id));
    }
    return tokenizerIds;
}

} // namespace

/**
 * @brief Demo stage: trains a Unigram tokenizer on a repeated sentence, then
 * builds a GPT with Mixture-of-Experts layers over resulting vocabulary
 * and generates text from a prompt.
 *
 * @remark The final check prints a warning if the generated text contains
 * neither "brown" nor "fox" from the training sentence.
 *
 * @return 0 on completion.
 */
int testGPTWithUnigram() {
    // 1. Data
    // We'll use a longer repeating text to allow patterns to emerge.
    // "The quick brown fox jumps over lazy dog."
    std::string text = "the quick brown fox jumps over lazy dog ";
    // Repeat it
    std::string corpus;
    for (int i = 0; i < 50; i++) {
        corpus += text;
    }

    // 2. Train Tokenizer
    std::cout << "Training Unigram Tokenizer..." << std::endl;
    // Small vocab for small data
    UnigramTokenizer tokenizer(100);
    // 20 iters
    tokenizer.train(corpus, 20);

    std::vector<int> tokens = tokenizer.encode(corpus);
    std::cout << "Corpus encoded to " 
        << tokens.size() << " tokens." << std::endl;
    std::cout << "Vocab size: " 
        << tokenizer.vocab.size() << std::endl;

    // Show some samples
    std::cout << "Sample tokens: ";
    for (std::size_t i = 0; i < 10 && i < tokens.size(); i++) {
        std::cout << tokens[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Sample decode: " << 
        tokenizer.decode({ tokens.begin(), tokens.begin() + 10 }) 
        << std::endl;

    // 3. Train GPT: dModel and ctxLen must match GPTWithUnigramConfig
    // (d_head = 32, maxSeqLen = 32), which sizes RotaryEmbedding.
    std::size_t vocabSize = tokenizer.vocab.size();
    std::size_t dModel = 32;
    std::size_t numLayers = 2;
    std::size_t ctxLen = 32;
    bool useMoe = true;
    std::size_t numExperts = 4;
    std::size_t topK = 2;

    GPTWithUnigram gpt(vocabSize, dModel, numLayers, ctxLen, useMoe, numExperts, topK);

    std::cout << "Training GPT (MoE enabled)..." << std::endl;
    double learningRate = 0.01;

    // The model takes size_t token ids; the tokenizer produces int ids.
    std::vector<std::size_t> tokenIds = toModelIds(tokens);

    for (int epoch = 0; epoch < 200; epoch++) {
        double totalLoss = 0;
        int stepCount = 0;

        // Sliding window: non-overlapping ctxLen-token chunks; target for
        // each token is token that follows it.
        for (std::size_t i = 0; i < tokenIds.size() - ctxLen - 1; i += ctxLen) {
            std::vector<std::size_t> inputIds, targetIds;
            for (std::size_t j = 0; j < ctxLen; j++) {
                inputIds.push_back(tokenIds[i + j]);
                targetIds.push_back(tokenIds[i + j + 1]);
            }
            totalLoss += gpt.trainStep(inputIds, targetIds, learningRate);
            stepCount++;
        }

        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch 
                << " Loss: " << (stepCount ? totalLoss / stepCount : 0) 
                << std::endl;
        }
    }

    // 4. Generate: encode prompt with tokenizer, let model
    // continue it, and decode ids back to text.
    std::string prompt = "the quick";
    std::vector<int> promptIds = tokenizer.encode(prompt);
    std::cout << "Prompt: '" << prompt << "' -> IDs: ";
    for (auto tokenId : promptIds) {
        std::cout << tokenId << " ";
    }
    std::cout << std::endl;

    // Convert tokenizer ids to the size_t ids DecoderOnlyModel expects, and
    // the generated ids back for decoding.
    std::vector<std::size_t> promptTokens = toModelIds(promptIds);
    // Gen 20 tokens
    std::vector<std::size_t> generatedTokens = gpt.generate(promptTokens, 20);
    std::string generatedText = tokenizer.decode(toTokenizerIds(generatedTokens));

    std::cout << "Generated: [" << generatedText << "]" << std::endl;

    if (std::string::npos != generatedText.find("brown") || 
        generatedText.find("fox") != std::string::npos) {
        std::cout << "Test Passed: Generated relevant text." << std::endl;
    }
    else {
        std::cout << "Test Warning: Generated text might "
            "be gibberish (needs more training/data)." << std::endl;
    }
    return 0;
}