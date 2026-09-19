#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include "BERT.h"

// Anonymous namespace: keeps these helpers local to this translation unit,
// since TestWordPieceTokenizer.cpp declares its own same-named getId/
// Example/generateExample for its own vocab.
namespace {

// Simple Mock Tokenizer: a fixed word list, with ids given by position.
std::map<std::string, std::size_t> wordToId;
std::vector<std::string> idToWord;

/**
 * @brief Fills wordToId/idToWord with special tokens ([PAD], [UNK], [CLS],
 * [SEP], [MASK]) followed by words used in toy corpus.
 */
void buildVocab() {
    std::vector<std::string> words = {
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "the", "cat", "dog", "ate", "food", "sat", "on", "mat", "ran", "fast", "heavy", "rain"
    };
    for (const auto& word : words) {
        wordToId[word] = idToWord.size();
        idToWord.push_back(word);
    }
}

/**
 * @brief Looks up a word's id; unknown words map to [UNK].
 */
std::size_t getId(const std::string& word) {
    if (wordToId.count(word)) {
        return wordToId[word];
    }
    return wordToId["[UNK]"];
}

/**
 * @brief One BERT training example: a masked two-sentence input plus its
 * MLM and NSP targets.
 */
class Example {
public:
    std::vector<std::size_t> inputIds;      // Input IDs (masked)
    std::vector<std::size_t> typeIds;    // Segment IDs
    std::vector<std::size_t> mlmLabels; // Original IDs for masked pos, 0 otherwise
    std::size_t nspLabel;             // 0 or 1
};

/**
 * @brief Builds one random training example from toy corpus:
 * [CLS] sentence A [SEP] sentence B [SEP], where B is either same
 * sentence as A (label 0, "IsNext") or a different one (label 1), with one
 * random non-special token replaced by [MASK].
 *
 * @param randomEngine Random engine; advances as sentences and mask are picked.
 */
Example generateExample(std::mt19937& randomEngine) {
    std::vector<std::vector<std::string>> corpus = {
        {"the", "cat", "sat", "on", "mat"},
        {"the", "dog", "ran", "fast"},
        {"cat", "ate", "food"},
        {"dog", "ate", "food"},
        {"heavy", "rain", "fast"}
    };

    std::uniform_int_distribution<std::size_t> sentenceDist(0, corpus.size() - 1);
    std::uniform_int_distribution<std::size_t> coinFlip(0, 1);

    // Pick A
    std::size_t indexA = sentenceDist(randomEngine);
    std::vector<std::string> sentenceA = corpus[indexA];

    // Pick B
    std::size_t indexB;
    // NSP label convention: 0 = IsNext, 1 = Random (NotNext). Since this
    // corpus is just a bag of unrelated sentences with no real "next
    // sentence" relationship, IsNext is simulated by pairing a sentence
    // with itself.
    std::size_t nspLabel = 0;

    if (coinFlip(randomEngine) == 0) {
        // Positive Pair (Simulated by picking same sentence or related)
        // "Next" sentence is same as A for simplicity in this tiny corpus
        indexB = indexA; 
        nspLabel = 0;
    }
    else {
        // Negative Pair
        indexB = sentenceDist(randomEngine);
        while (indexB == indexA) {
            indexB = sentenceDist(randomEngine);
        }
        nspLabel = 1;
    }

    std::vector<std::string> sentenceB = corpus[indexB];

    // Construct Input
    // [CLS] A [SEP] B [SEP]
    std::vector<std::size_t> inputIds;
    std::vector<std::size_t> typeIds;
    std::vector<std::size_t> mlmLabels;

    inputIds.push_back(getId("[CLS]"));
    typeIds.push_back(0);
    mlmLabels.push_back(0); // Ignore

    for (const auto& word : sentenceA) {
        inputIds.push_back(getId(word));
        typeIds.push_back(0);
        mlmLabels.push_back(0);
    }
    inputIds.push_back(getId("[SEP]"));
    typeIds.push_back(0);
    mlmLabels.push_back(0);

    for (const auto& word : sentenceB) {
        inputIds.push_back(getId(word));
        typeIds.push_back(1);
        mlmLabels.push_back(0);
    }
    inputIds.push_back(getId("[SEP]"));
    typeIds.push_back(1);
    mlmLabels.push_back(0);

    // MASKING (as in BERT's masked-language-model objective): hide one random
    // ordinary token so the model must recover it from both sides' context.
    // Pick 1 random token to mask (excluding specials)
    std::vector<std::size_t> candidateIndices;
    for (std::size_t i = 0; i < inputIds.size(); i++) {
        std::size_t tokenId = inputIds[i];
        if (tokenId != getId("[CLS]") && tokenId != getId("[SEP]") && tokenId != getId("[PAD]")) {
            candidateIndices.push_back(i);
        }
    }

    if (!candidateIndices.empty()) {
        std::uniform_int_distribution<std::size_t> maskDist(0, candidateIndices.size() - 1);
        std::size_t maskIdx = candidateIndices[maskDist(randomEngine)];

        // Save label
        mlmLabels[maskIdx] = inputIds[maskIdx];

        // Apply Mask
        inputIds[maskIdx] = getId("[MASK]");
    }

    return { inputIds, typeIds, mlmLabels, nspLabel };
}

} // namespace

/**
 * @brief Demo stage: trains a small BERT on toy corpus (masked-word
 * prediction and next-sentence prediction), then checks one masked-word
 * prediction and one next-sentence prediction.
 *
 * @return 0 on completion.
 */
int testBert() {
    std::cout << "Building Vocab..." << std::endl;
    buildVocab();
    std::size_t vocabSize = idToWord.size();
    std::cout << "Vocab Size: " << vocabSize << std::endl;

    // Model Config
    std::size_t dModel = 32;
    std::size_t numLayers = 2; // Small for testing
    std::size_t maxLen = 50;

    std::cout << "Initializing BERT..." << std::endl;
    BertModel<double> bert(vocabSize, dModel, numLayers, maxLen);

    // Fixed seed: the training examples are the same on every run.
    std::mt19937 randomEngine(42);

    std::cout << "Starting Training Loop..." << std::endl;
    double learningRate = 0.001; // Careful with LR

    for (int epoch = 0; epoch < 40; epoch++) {
        double epochLoss = 0;
        int stepsPerEpoch = 20;

        // Each step trains on one freshly generated example. The Adam timestep
        // must be 1-based and keep increasing across epochs.
        for (int stepIndex = 0; stepIndex < stepsPerEpoch; stepIndex++) {
            Example example = generateExample(randomEngine);
            double loss = bert.trainStep(example.inputIds, example.typeIds, example.mlmLabels, example.nspLabel, learningRate, UpdateRule::adam(epoch * stepsPerEpoch + stepIndex + 1));
            epochLoss += loss;
        }

        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << epochLoss / stepsPerEpoch << std::endl;
        }
    }

    std::cout << "Training Complete." << std::endl;

    // Verification Inference
    std::cout << "\nInference Test:" << std::endl;
    // "the cat [MASK] on mat" -> Expect "sat"
    std::vector<std::string> testSent = { "[CLS]", "the", "cat", "[MASK]", "on", "mat", "[SEP]" };
    std::vector<std::size_t> inputIds;
    std::vector<std::size_t> typeIds;
    for (auto& word : testSent) {
        inputIds.push_back(getId(word));
        typeIds.push_back(0);
    }

    Tensor<double> logits;
    bert.predictMaskedLogits(inputIds, typeIds, logits);

    // Find the model's prediction at the [MASK] position (index 3 in the test
    // sentence): score every vocabulary word there and take the highest.
    std::size_t maskPos = 3; // "the cat [MASK]" -> 0 1 2 3

    double bestScore = -1e9;
    std::size_t predictedId = 0;

    std::size_t maskRowOffset = maskPos * vocabSize;
    for (std::size_t tokenId = 0; tokenId < vocabSize; tokenId++) {
        if (logits.data[maskRowOffset + tokenId] > bestScore) {
            bestScore = logits.data[maskRowOffset + tokenId];
            predictedId = tokenId;
        }
    }

    std::cout << "Input: cat [MASK] on mat" << std::endl;
    std::cout << "Predicted: " << idToWord[predictedId] << std::endl;

    if (idToWord[predictedId] == "sat") {
        std::cout << "SUCCESS: Correctly predicted 'sat'" << std::endl;
    }
    else {
        std::cout << "FAILURE: Predicted '" 
            << idToWord[predictedId] << "' instead of 'sat'" << std::endl;
    }

    // NSP Test: training labelled a pair 0 (IsNext) only when sentence B was
    // identical to sentence A, so test with same sentence twice; the
    // model should score IsNext (index 0) higher.

    std::vector<std::string> nspSent = 
        { "[CLS]", "cat", "ate", "food", "[SEP]", "cat", "ate", "food", "[SEP]" };
    
    inputIds.clear(); typeIds.clear();
    for (auto& word : nspSent) {
        inputIds.push_back(getId(word));
        // Segment ids: positions 0-4 ([CLS] A [SEP]) are segment 0, the
        // rest (B [SEP]) are segment 1.
        if (inputIds.size() <= 5) {
            typeIds.push_back(0);
        }
        else {
            typeIds.push_back(1);
        }
    }

    Tensor<double> encoded;
    bert.forwardEncoder(inputIds, typeIds, encoded);

    // Pooler: NSP head reads encoded [CLS] vector (position 0).
    Tensor<double> clsToken({ 1, dModel });
    for (std::size_t featureIdx = 0; featureIdx < dModel; featureIdx++) {
        clsToken[featureIdx] = encoded.data[featureIdx];
    }

    Tensor<double> nspLogits;
    bert.nspProj.forward(clsToken, nspLogits);

    std::cout << "NSP Logits (IsNext vs NotNext): " << nspLogits[0] << " / " << nspLogits[1] << std::endl;
    if (nspLogits[0] > nspLogits[1]) {
        std::cout << "NSP Prediction: IsNext (Correct)" << std::endl;
    }
    else {
        std::cout << "NSP Prediction: NotNext" << std::endl;
    }

    return 0;
}