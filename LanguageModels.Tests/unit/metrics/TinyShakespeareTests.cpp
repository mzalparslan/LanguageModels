#include "pch.h"
#include "NGramBaseline.h"
#include "TestSupport.h"
#include "VanillaRNN.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

// Tiny Shakespeare (resources/tinyshakespeare.txt): concatenated Shakespeare
// text from Andrej Karpathy's char-rnn, standard small benchmark for
// character-level language models. Its size and alphabet are widely published
// (1,115,394 characters, 65 distinct, 40,000 lines), which lets these tests
// confirm right data is loaded before any model is judged on it.
//
// expected baseline scores were computed with an independent Python
// implementation, so they check NGramBaseline and Metrics against numbers not
// produced by this library.

namespace {
    struct Corpus {
        std::string text;
        std::vector<std::size_t> ids;   // each byte's rank among distinct bytes
        std::vector<int> intIds;        // same ids, for VanillaRNN's int interface
        std::size_t vocab = 0;
    };

    const Corpus& corpus() {
        static const Corpus loaded = [] {
            Corpus result;
            result.text = testsupport::readResource("tinyshakespeare.txt");

            std::set<unsigned char> alphabet(result.text.begin(), result.text.end());
            std::map<unsigned char, std::size_t> idOf;
            for (unsigned char byte : alphabet) {
                std::size_t next = idOf.size();
                idOf[byte] = next;
            }
            for (unsigned char byte : result.text) {
                result.ids.push_back(idOf[byte]);
                result.intIds.push_back(static_cast<int>(idOf[byte]));
            }
            result.vocab = alphabet.size();
            return result;
        }();
        return loaded;
    }

    std::vector<std::size_t> slice(const std::vector<std::size_t>& ids, std::size_t begin, std::size_t end) {
        return std::vector<std::size_t>(ids.begin() + begin, ids.begin() + end);
    }
}

TEST(TinyShakespeareTest, MatchesThePublishedSizeAndAlphabet) {
    const Corpus& data = corpus();
    ASSERT_FALSE(data.text.empty()) << "resources/tinyshakespeare.txt was not found";

    EXPECT_EQ(data.text.size(), 1115394u);
    EXPECT_EQ(data.vocab, 65u);
    EXPECT_EQ(std::count(data.text.begin(), data.text.end(), '\n'), 40000);
    EXPECT_EQ(data.text.compare(0, 14, "First Citizen:"), 0);
}

TEST(TinyShakespeareTest, UnigramBaselineMatchesTheEntropyOfTheText) {
    const Corpus& data = corpus();
    ASSERT_FALSE(data.ids.empty());
    NGramBaseline unigram(data.vocab, 1, 0.0);
    unigram.train(data.ids);

    Metrics metrics = unigram.evaluate(data.ids);

    // Letter-frequency entropy of text: 4.779 bits, perplexity 27.46,
    // far below 65 of a uniform guess. most frequent character is space.
    EXPECT_NEAR(metrics.loss, 3.31279245, 1e-6);
    EXPECT_NEAR(metrics.perplexity, 27.4617039, 1e-4);
    EXPECT_NEAR(metrics.bpc, 4.7793, 1e-3);
    EXPECT_NEAR(metrics.accuracy, 0.152316, 1e-6);
    EXPECT_LT(metrics.perplexity, static_cast<double>(data.vocab));
}

TEST(TinyShakespeareTest, BigramBaselineMatchesTheConditionalEntropyOfTheText) {
    const Corpus& data = corpus();
    ASSERT_FALSE(data.ids.empty());
    NGramBaseline bigram(data.vocab, 2, 0.0);
    bigram.train(data.ids);

    Metrics metrics = bigram.evaluate(data.ids);

    // One character of context roughly halves uncertainty: perplexity 11.6.
    EXPECT_NEAR(metrics.loss, 2.4525654, 1e-6);
    EXPECT_NEAR(metrics.perplexity, 11.6181136, 1e-4);
    EXPECT_NEAR(metrics.accuracy, 0.271337, 1e-6);
}

TEST(TinyShakespeareTest, HeldOutBaselinesFollowTheUsualNinetyTenSplit) {
    const Corpus& data = corpus();
    ASSERT_FALSE(data.ids.empty());
    const std::size_t cut = data.ids.size() * 9 / 10;
    auto train = slice(data.ids, 0, cut);
    auto validation = slice(data.ids, cut, data.ids.size());

    NGramBaseline unigram(data.vocab, 1, 1.0);
    NGramBaseline bigram(data.vocab, 2, 1.0);
    unigram.train(train);
    bigram.train(train);

    // Laplace smoothing (add 1) on last 10% of text.
    EXPECT_NEAR(unigram.evaluate(validation).perplexity, 28.4260317, 1e-3);
    EXPECT_NEAR(bigram.evaluate(validation).perplexity, 11.9638480, 1e-3);
}

TEST(TinyShakespeareTest, AnUntrainedCharacterRnnStartsNearAUniformGuess) {
    const Corpus& data = corpus();
    ASSERT_FALSE(data.ids.empty());
    std::vector<int> inputs(data.intIds.begin(), data.intIds.begin() + 200);
    std::vector<int> targets(data.intIds.begin() + 1, data.intIds.begin() + 201);
    VanillaRNN<double> rnn(32, data.vocab);

    Metrics metrics = rnn.evaluate(inputs, targets, rnn.getZeroState());

    EXPECT_NEAR(metrics.perplexity, static_cast<double>(data.vocab), 3.0);
}

TEST(TinyShakespeareTest, ACharacterRnnLearnsBeyondTheUniformGuessAndGeneralises) {
    const Corpus& data = corpus();
    ASSERT_FALSE(data.ids.empty());
    const std::size_t cut = data.ids.size() * 9 / 10;
    VanillaRNN<double> rnn(48, data.vocab);

    // A short run over start of training split, in windows of 25 characters.
    // (Learning rate 0.01: loss is summed over window, so 0.1 diverges here.)
    const std::size_t window = 25;
    auto hidden = rnn.getZeroState();
    for (int step = 0; step < 400; step++) {
        std::size_t start = static_cast<std::size_t>(step) * window;
        std::vector<int> inputs(data.intIds.begin() + start, data.intIds.begin() + start + window);
        std::vector<int> targets(data.intIds.begin() + start + 1, data.intIds.begin() + start + window + 1);
        (void)rnn.trainStep(inputs, targets, hidden, 0.01);
    }

    // Judge on text model never trained on.
    std::vector<int> inputs(data.intIds.begin() + cut, data.intIds.begin() + cut + 500);
    std::vector<int> targets(data.intIds.begin() + cut + 1, data.intIds.begin() + cut + 501);
    Metrics metrics = rnn.evaluate(inputs, targets, rnn.getZeroState());

    // Deterministic (fixed seed): about 32 after this run, against 65 for a uniform guess.
    EXPECT_LT(metrics.perplexity, 0.6 * static_cast<double>(data.vocab));
}
