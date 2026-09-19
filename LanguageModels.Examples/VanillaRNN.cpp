#include "VanillaRNN.h"
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <thread>
#include <chrono>

/**
 * @brief Demo stage: trains a character-level VanillaRNN on a short repeating
 * string ("hello world" x4) and periodically prints loss/perplexity/accuracy,
 * model's predictions for current window, and a sampled string.
 *
 * @return 0 on completion.
 */
int testVanilla() {
    // 1. Prepare Data
    std::string data = "hello world hello world hello world hello world";
    // Unique chars: vocabulary is set of characters in data;
    // each gets an integer id (its rank in sorted order).
    std::set<char> charSet(data.begin(), data.end());
    std::vector<char> chars(charSet.begin(), charSet.end());
    std::size_t dataSize = data.size();
    std::size_t vocabSize = chars.size();
    std::map<char, int> charToIndex;
    std::map<int, char> indexToChar;

    std::cout << "Data: " << data << "\n";
    std::cout << "Vocab Size: " << vocabSize << "\n";

    int nextId = 0;
    for (char character : chars) {
        charToIndex[character] = nextId;
        indexToChar[nextId] = character;
        nextId++;
    }
    // 2. Training Setup: hiddenSize is state width, seqLength is how
    // many steps are unrolled (and backpropagated through) per update.
    // Small network
    std::size_t hiddenSize = 32;
    // Steps unrolled
    std::size_t seqLength = 5;
    double learningRate = 0.1;

    VanillaRNN<double> rnn(hiddenSize, vocabSize);
    // data pointer: start of current window in text
    std::size_t windowStart = 0; 
    std::vector<double> hiddenPrev = rnn.getZeroState();

    // Train for 101 iterations, sliding a window over the text (about 10 passes).
    for (int iteration = 0; iteration < 101; ++iteration) {
        // Prepare inputs (X) and targets (Y): target for each character
        // is character that follows it. When window would run past the
        // end of text, wrap to start and reset hidden state.
        if (windowStart + seqLength + 1 >= dataSize) {
            windowStart = 0; // reset
            hiddenPrev = rnn.getZeroState(); // reset state
        }
        std::vector<int> inputs;
        std::vector<int> targets;
        for (std::size_t step = 0; step < seqLength; ++step) {
            inputs.push_back(charToIndex[data[windowStart + step]]);
            targets.push_back(charToIndex[data[windowStart + step + 1]]);
        }
        // Backup state for logging
        std::vector<double> hiddenPrevBackup = hiddenPrev;
        // Forward + Backward + Update (hiddenPrev is advanced to final state)
        double loss = rnn.trainStep(inputs, targets, hiddenPrev, learningRate);

        // Every 10 iterations: report the metrics and show what the model predicts.
        if (iteration % 10 == 0) {
            auto metrics = rnn.evaluate(inputs, targets, hiddenPrevBackup);
            std::cout << "Iter " << iteration
                << ", Loss: " << metrics.loss
                << ", PPL: " << metrics.perplexity
                << ", Acc: " << metrics.accuracy * 100 << "%"
                << ", BPC: " << metrics.bpc << "\n";
            // Show Input vs Target vs Pred
            std::string inputStr = "";
            std::string targetStr = "";
            std::string predStr = "";

            // Re-run forward to get current predictions for this input
            // Use backup state so we see what network actually thought *before* update
            auto cache = rnn.forward(inputs, hiddenPrevBackup);

            for (int step = 0; step < seqLength; step++) {
                inputStr += indexToChar[inputs[step]];
                targetStr += indexToChar[targets[step]];

                // Greedy prediction
                int bestIndex = 0;
                double bestProb = -1;
                for (int charId = 0; charId < vocabSize; charId++) {
                    if (cache.probabilities[step][charId] > bestProb) {
                        bestProb = cache.probabilities[step][charId];
                        bestIndex = charId;
                    }
                }
                predStr += indexToChar[bestIndex];
            }

            std::cout << "Input:  [" << inputStr << "]\n";
            std::cout << "Target: [" << targetStr << "]\n";
            std::cout << "Pred:   [" << predStr << "]\n";
            // Sample: let the model generate 20 characters from its current state.
            std::vector<std::size_t> sampleIndices = rnn.sample(hiddenPrev, inputs[0], 20);
            std::string sampleText = "";
            for (std::size_t charIndex : sampleIndices) {
                sampleText += indexToChar[static_cast<int>(charIndex)];
            }
            std::cout << "Sample: " << sampleText << "\n-----------------\n";
        }

        windowStart += seqLength;
    }
    std::cout << "Done.\n";
    return 0;
}

