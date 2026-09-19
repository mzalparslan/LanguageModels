
#include "MiniTransformer.h"

#include <iostream>
#include <map>
#include <string>
#include <sstream>

/**
 * @brief Minimal word-level tokenizer: every distinct whitespace-separated
 * word gets next free id, in order of first appearance.
 */
class SimpleTokenizer {
public:
    // Word -> id.
    std::map<std::string, std::size_t> wordToId;
    // Id -> word (the inverse of wordToId).
    std::map<std::size_t, std::string> idToWord;
    // Number of distinct words seen so far.
    std::size_t vocabSize = 0;

    /**
     * @brief Adds any words of sentence that are not in vocabulary yet.
     */
    void add(const std::string& sentence) {
        std::stringstream stream(sentence);
        std::string word;
        while (stream >> word) {
            if (wordToId.find(word) == wordToId.end()) {
                wordToId[word] = vocabSize;
                idToWord[vocabSize] = word;
                vocabSize++;
            }
        }
    }

    /**
     * @brief Converts a sentence to word ids; unknown words map to id 0.
     */
    std::vector<std::size_t> encode(const std::string& sentence) {
        std::stringstream stream(sentence);
        std::string word;
        std::vector<std::size_t> ids;
        while (stream >> word) {
            if (wordToId.find(word) != wordToId.end()) {
                ids.push_back(wordToId[word]);
            }
            else {
                ids.push_back(0); // UNK
            }
        }
        return ids;
    }

    /**
     * @brief Converts ids back to words joined by spaces; unknown ids are skipped.
     */
    std::string decode(const std::vector<std::size_t>& ids) {
        std::string text = "";
        for (std::size_t id : ids) {
            if (idToWord.count(id)) {
                text += idToWord[id] + " ";
            }
        }
        return text;
    }
};

/**
 * @brief Demo stage: trains MiniTransformer on four tiny English -> French
 * sentence pairs until it memorizes them, then translates same sentences
 * with greedy decoding.
 *
 * @return 0 on completion.
 */
int testSimpleSet() {
    // 1. Dataset
    std::vector<std::pair<std::string, std::string>> data = {
        {"i love you", "<SOS> je t'aime <EOS>"},
        {"hello world", "<SOS> bonjour le monde <EOS>"},
        {"good morning", "<SOS> bonjour <EOS>"},
        {"i love code", "<SOS> j'aime le code <EOS>"}
    };

    // 2. Tokenizers: one vocabulary per language, built from the dataset itself.
    SimpleTokenizer encoderTokenizer, decoderTokenizer;
    for (auto& sentencePair : data) {
        encoderTokenizer.add(sentencePair.first);
        decoderTokenizer.add(sentencePair.second);
    }

    std::cout << "Src Vocab: " << encoderTokenizer.vocabSize << "\n";
    std::cout << "Tgt Vocab: " << decoderTokenizer.vocabSize << "\n";

    // 3. Model: vocabulary sizes set the embedding tables and the output width.
    MiniTransformer<double> model(encoderTokenizer.vocabSize, decoderTokenizer.vocabSize);
    double learningRate = 0.005;

    // 4. Training Loop: one optimizer step per sentence pair (batch size 1).
    std::cout << "Training...\n";
    // Adam timestep: 1-based, increased after every training example.
    std::size_t adamStep = 1;
    for (std::size_t epoch = 0; epoch < 400; epoch++) {
        double totalLoss = 0;

        for (auto& sentencePair : data) {
            auto src = encoderTokenizer.encode(sentencePair.first);
            auto tgtFull = decoderTokenizer.encode(sentencePair.second);

            if (tgtFull.size() < 2) {
                continue;
            }

            // Teacher forcing: decoder is fed target minus its last
            // token and must predict target minus its first token, i.e.
            // at every position it predicts next word.
            std::vector<std::size_t> decoderInput(tgtFull.begin(), tgtFull.end() - 1);
            std::vector<std::size_t> expectedIds(tgtFull.begin() + 1, tgtFull.end());
            // (For "<SOS> je t'aime <EOS>": input "<SOS> je t'aime", expected "je t'aime <EOS>".)

            totalLoss += model.trainStep(src, decoderInput, expectedIds, learningRate, UpdateRule::adam(adamStep));
            adamStep++;
        }

        // Report progress rarely: this dataset trains in well under a second per epoch.
        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << " Loss: " << totalLoss << "\n";
            // Less aggressive decay
            // lr *= 0.99; 
        }
    }

    // 5. Test Inference (Greedy)
    std::cout << "\nInference:\n";

    for (auto& sentencePair : data) {
        std::cout << "Src: " << sentencePair.first << " -> ";
        std::vector<std::size_t> src = encoderTokenizer.encode(sentencePair.first);

        // Start with <SOS> and grow target one word at a time, feeding
        // each prediction back in, until <EOS> or 10 words.
        std::vector<std::size_t> tgt = { decoderTokenizer.wordToId["<SOS>"] };

        for (std::size_t step = 0; step < 10; ++step) {
            Tensor<double> logits;
            model.forward(src, tgt, logits);

            // Get last token logits: only last position predicts word
            // that comes next.
            std::size_t seqLen = logits.shape[0];
            std::size_t outputVocabSize = logits.shape[1];

            // Arg-max over the vocabulary: the word with the highest score wins.
            std::size_t bestId = 0;
            double maxVal = -1e9;
            for (std::size_t wordId = 0; wordId < outputVocabSize; wordId++) {
                double score = logits.data[(seqLen - 1) * outputVocabSize + wordId];
                if (score > maxVal) {
                    maxVal = score;
                    bestId = wordId;
                }
            }

            if (decoderTokenizer.idToWord[bestId] == "<EOS>") {
                break;
            }
            tgt.push_back(bestId);
        }

        std::cout << decoderTokenizer.decode(tgt) << "\n";
    }

    return 0;
}