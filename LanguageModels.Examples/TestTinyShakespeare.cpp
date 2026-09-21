#include "NGramBaseline.h"
#include "VanillaRNN.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Set by --quick in Main.cpp.
extern bool useQuickDataset;

namespace {
    void printRow(const std::string& name, const Metrics& metrics, bool showAccuracy = true) {
        std::cout << std::left << std::setw(28) << name << std::right << std::fixed
            << std::setprecision(3)
            << std::setw(8) << metrics.loss
            << std::setw(12) << metrics.perplexity
            << std::setw(10) << metrics.bpc
            << std::setw(10);
        if (showAccuracy) {
            std::cout << metrics.accuracy * 100 << "%\n";
        }
        else {
            std::cout << "n/a" << "\n";
        }
    }

    // Scores a long text in windows (a recurrent model's forward pass keeps one
    // entry per step, so 100k steps at once is wasteful), pooling totals.
    Metrics evaluateInWindows(VanillaRNN<double>& rnn, const std::vector<int>& ids,
        std::size_t chars, std::size_t window) {
        double totalLoss = 0;
        std::size_t correct = 0;
        std::size_t count = 0;
        for (std::size_t start = 0; start + 1 < chars; start += window) {
            std::size_t length = std::min(window, chars - 1 - start);
            std::vector<int> inputs(ids.begin() + start, ids.begin() + start + length);
            std::vector<int> targets(ids.begin() + start + 1, ids.begin() + start + length + 1);
            Metrics part = rnn.evaluate(inputs, targets, rnn.getZeroState());
            totalLoss += part.loss * length;
            correct += static_cast<std::size_t>(std::llround(part.accuracy * length));
            count += length;
        }
        return Metrics::fromTotals(totalLoss, correct, count);
    }
}

/**
 * @brief Demo stage: standard character-level benchmark. Loads Tiny
 * Shakespeare, checks it against its published size and alphabet, then scores
 * a character RNN on held-out text next to baselines whose perplexity is
 * known: a uniform guess, letter frequencies (unigram) and one character of
 * context (bigram). A model worth keeping has to beat baselines.
 *
 * @return 0 on completion, 1 if data file is missing.
 */
int testTinyShakespeare() {
    std::ifstream file("tinyshakespeare.txt", std::ios::binary);
    if (!file) {
        std::cout << "tinyshakespeare.txt not found next to executable (it lives in resources/).\n";
        return 1;
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    const std::string text = contents.str();

    // Each distinct character gets an id (its rank in sorted order).
    std::set<unsigned char> alphabet(text.begin(), text.end());
    std::map<unsigned char, int> idOf;
    for (unsigned char byte : alphabet) {
        int next = static_cast<int>(idOf.size());
        idOf[byte] = next;
    }
    std::vector<int> ids;
    for (unsigned char byte : text) {
        ids.push_back(idOf[byte]);
    }
    const std::size_t vocab = alphabet.size();

    std::cout << "Characters: " << text.size() << " (published: 1115394)\n";
    std::cout << "Vocabulary: " << vocab << " (published: 65)\n";

    // 90% train / 10% validation, split by position (the usual arrangement).
    const std::size_t cut = ids.size() * 9 / 10;
    const std::size_t evalChars = useQuickDataset ? 20000 : ids.size() - cut;
    std::vector<int> validation(ids.begin() + cut, ids.begin() + cut + evalChars);

    // Baselines: counted from training part, scored on validation part.
    std::vector<std::size_t> trainIds(ids.begin(), ids.begin() + cut);
    std::vector<std::size_t> validationIds(validation.begin(), validation.end());
    NGramBaseline unigram(vocab, 1, 1.0);
    NGramBaseline bigram(vocab, 2, 1.0);
    unigram.train(trainIds);
    bigram.train(trainIds);

    // Train character RNN with plain SGD in windows of 25 characters,
    // carrying hidden state from window to window.
    const std::size_t hiddenSize = 64;
    const std::size_t window = 25;
    const double learningRate = 0.01;
    // full run makes about one pass over training text (50000 x 25 characters).
    const int iterations = useQuickDataset ? 2000 : 50000;
    VanillaRNN<double> rnn(hiddenSize, vocab);
    std::vector<double> hidden = rnn.getZeroState();

    std::cout << "\nTraining a " << hiddenSize << "-unit RNN for " << iterations
        << " steps of " << window << " characters"
        << (useQuickDataset ? " (--quick)" : "") << "...\n";
    auto begin = std::chrono::steady_clock::now();
    std::size_t position = 0;
    for (int iteration = 1; iteration <= iterations; iteration++) {
        if (position + window + 1 >= cut) {
            position = 0;
            hidden = rnn.getZeroState();
        }
        std::vector<int> inputs(ids.begin() + position, ids.begin() + position + window);
        std::vector<int> targets(ids.begin() + position + 1, ids.begin() + position + window + 1);
        (void)rnn.trainStep(inputs, targets, hidden, learningRate);
        position += window;

        if (iteration % (iterations / 5) == 0) {
            Metrics progress = evaluateInWindows(rnn, validation, std::min<std::size_t>(evalChars, 5000), 1000);
            std::cout << "  step " << std::setw(6) << iteration << "  validation perplexity "
                << std::fixed << std::setprecision(2) << progress.perplexity << "\n";
        }
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    std::cout << "Elapsed: " << std::fixed << std::setprecision(1) << seconds << " s\n";

    Metrics rnnMetrics = evaluateInWindows(rnn, validation, evalChars, 1000);
    Metrics unigramMetrics = unigram.evaluate(validationIds);
    Metrics bigramMetrics = bigram.evaluate(validationIds);

    std::cout << "\nValidation (" << evalChars << " characters)\n";
    std::cout << std::left << std::setw(28) << "Model" << std::right
        << std::setw(8) << "loss" << std::setw(12) << "perplexity"
        << std::setw(10) << "bits/char" << std::setw(11) << "accuracy" << "\n";
    Metrics uniform = Metrics::fromTotals(std::log(static_cast<double>(vocab)) * 1000, 0, 1000);
    printRow("Uniform guess (= vocab size)", uniform, false);
    printRow("Unigram (letter frequency)", unigramMetrics);
    printRow("Bigram (1 char of context)", bigramMetrics);
    printRow("Character RNN (trained here)", rnnMetrics);

    std::cout << "\nRNN beats uniform guess: " << (rnnMetrics.perplexity < uniform.perplexity ? "yes" : "NO")
        << "\nRNN beats unigram baseline: " << (rnnMetrics.perplexity < unigramMetrics.perplexity ? "yes" : "no")
        << "\nRNN beats bigram baseline: " << (rnnMetrics.perplexity < bigramMetrics.perplexity ? "yes" : "no")
        << "\n";
    return 0;
}
