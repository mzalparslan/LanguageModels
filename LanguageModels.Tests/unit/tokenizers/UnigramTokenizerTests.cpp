#include "pch.h"
#include "UnigramTokenizer.h"
#include "TestSupport.h"
#include <cmath>

namespace {
    using Counts = std::unordered_map<std::string, std::size_t>;

    // A tokenizer whose vocabulary is set directly from counts, so
    // segmentation can be tested without running training loop.
    UnigramTokenizer makeTokenizer(const Counts& counts) {
        UnigramTokenizer tokenizer(1000);
        tokenizer.updateScores(counts);
        return tokenizer;
    }
}

// === construction

TEST(UnigramTokenizerTest, TargetVocabularySizeIsStored) {
    UnigramTokenizer tokenizer(50);

    EXPECT_EQ(tokenizer.vocabSizeTarget, 50u);
    EXPECT_TRUE(tokenizer.vocab.empty());
}

TEST(UnigramTokenizerTest, DefaultTargetIsOneThousand) {
    UnigramTokenizer tokenizer;

    EXPECT_EQ(tokenizer.vocabSizeTarget, 1000u);
}

TEST(UnigramTokenizerTest, ZeroTargetThrowsInvalidParameterSizeError) {
    EXPECT_THROW(UnigramTokenizer(0), InvalidParameterSizeError);
}

// -------------------------------------------------------------- updateScores

TEST(UnigramTokenizerScoresTest, IdsAreAssignedByDescendingFrequency) {
    auto tokenizer = makeTokenizer({ { "a", 6 }, { "b", 3 }, { "c", 1 } });

    EXPECT_EQ(tokenizer.vocab["a"].id, 0);
    EXPECT_EQ(tokenizer.vocab["b"].id, 1);
    EXPECT_EQ(tokenizer.vocab["c"].id, 2);
    EXPECT_EQ(tokenizer.idToToken, (std::vector<std::string>{ "a", "b", "c" }));
}

TEST(UnigramTokenizerScoresTest, ScoresAreLogProbabilities) {
    auto tokenizer = makeTokenizer({ { "a", 6 }, { "b", 3 }, { "c", 1 } });

    EXPECT_NEAR(tokenizer.vocab["a"].score, std::log(0.6), 1e-12);
    EXPECT_NEAR(tokenizer.vocab["b"].score, std::log(0.3), 1e-12);
    EXPECT_NEAR(tokenizer.vocab["c"].score, std::log(0.1), 1e-12);
}

TEST(UnigramTokenizerScoresTest, ProbabilitiesSumToOne) {
    auto tokenizer = makeTokenizer({ { "a", 6 }, { "b", 3 }, { "c", 1 }, { "de", 5 } });

    double total = 0.0;
    for (const auto& entry : tokenizer.vocab) {
        total += std::exp(entry.second.score);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(UnigramTokenizerScoresTest, ReplacesThePreviousVocabulary) {
    auto tokenizer = makeTokenizer({ { "a", 6 }, { "b", 3 } });

    tokenizer.updateScores({ { "x", 2 } });

    EXPECT_EQ(tokenizer.vocab.size(), 1u);
    EXPECT_EQ(tokenizer.idToToken, (std::vector<std::string>{ "x" }));
}

TEST(UnigramTokenizerScoresTest, EmptyCountsThrowInvalidSizeError) {
    UnigramTokenizer tokenizer;

    EXPECT_THROW(tokenizer.updateScores({}), InvalidSizeError);
}

TEST(UnigramTokenizerScoresTest, AllZeroCountsThrowDivisionByZeroError) {
    UnigramTokenizer tokenizer;

    EXPECT_THROW(tokenizer.updateScores({ { "a", 0 }, { "b", 0 } }), DivisionByZeroError);
}

// ------------------------------------------------------------- viterbiEncode

TEST(UnigramTokenizerViterbiTest, PrefersAFrequentMultiCharacterToken) {
    auto tokenizer = makeTokenizer({ { "a", 10 }, { "b", 10 }, { "ab", 30 } });

    EXPECT_EQ(tokenizer.viterbiEncode("ab"), (std::vector<std::string>{ "ab" }));
}

TEST(UnigramTokenizerViterbiTest, SplitsIntoCharactersWhenTheJoinedTokenIsRare) {
    auto tokenizer = makeTokenizer({ { "a", 10 }, { "b", 10 }, { "ab", 1 } });

    EXPECT_EQ(tokenizer.viterbiEncode("ab"), (std::vector<std::string>{ "a", "b" }));
}

TEST(UnigramTokenizerViterbiTest, ChoosesTheHighestScoringSegmentation) {
    auto tokenizer = makeTokenizer({ { "a", 10 }, { "b", 10 }, { "c", 10 },
        { "ab", 40 }, { "bc", 5 } });

    // "ab" + "c" (high score) beats "a" + "bc".
    EXPECT_EQ(tokenizer.viterbiEncode("abc"), (std::vector<std::string>{ "ab", "c" }));
}

TEST(UnigramTokenizerViterbiTest, TokensConcatenateBackToTheInput) {
    auto tokenizer = makeTokenizer({ { "a", 10 }, { "b", 10 }, { "c", 10 }, { "ab", 40 }, { "abc", 3 } });

    auto tokens = tokenizer.viterbiEncode("abcabcab");

    std::string joined;
    for (const auto& token : tokens) {
        joined += token;
    }
    EXPECT_EQ(joined, "abcabcab");
}

TEST(UnigramTokenizerViterbiTest, EmptyTextGivesNoTokens) {
    auto tokenizer = makeTokenizer({ { "a", 1 } });

    EXPECT_TRUE(tokenizer.viterbiEncode("").empty());
}

// ---------------------------------------------------------- encode / decode

TEST(UnigramTokenizerEncodeTest, EncodeMapsTokensToIds) {
    auto tokenizer = makeTokenizer({ { "a", 6 }, { "b", 3 }, { "c", 1 } });

    EXPECT_EQ(tokenizer.encode("abca"), (std::vector<int>{ 0, 1, 2, 0 }));
}

TEST(UnigramTokenizerEncodeTest, DecodeIsTheInverseOfEncode) {
    auto tokenizer = makeTokenizer({ { "a", 10 }, { "b", 10 }, { "c", 10 }, { "ab", 40 } });

    EXPECT_EQ(tokenizer.decode(tokenizer.encode("abcabc")), "abcabc");
}

TEST(UnigramTokenizerEncodeTest, EmptyTextEncodesToNoIds) {
    auto tokenizer = makeTokenizer({ { "a", 1 } });

    EXPECT_TRUE(tokenizer.encode("").empty());
    EXPECT_EQ(tokenizer.decode({}), "");
}

TEST(UnigramTokenizerEncodeTest, TextWithAnUnknownCharacterThrowsInvalidParameterError) {
    auto tokenizer = makeTokenizer({ { "a", 6 }, { "b", 3 } });

    EXPECT_THROW(tokenizer.encode("abz"), InvalidParameterError);
    EXPECT_THROW(tokenizer.encode("zab"), InvalidParameterError);
    EXPECT_THROW(tokenizer.encode("z"), InvalidParameterError);
}

TEST(UnigramTokenizerEncodeTest, EncodingWithAnEmptyVocabularyThrows) {
    UnigramTokenizer tokenizer;

    EXPECT_THROW(tokenizer.encode("abc"), InvalidParameterError);
}

TEST(UnigramTokenizerEncodeTest, DecodeRejectsIdsOutsideTheVocabulary) {
    auto tokenizer = makeTokenizer({ { "a", 6 }, { "b", 3 } });

    EXPECT_THROW(tokenizer.decode({ -1 }), InvalidParameterError);
    EXPECT_THROW(tokenizer.decode({ 0, 2 }), InvalidParameterError);
    EXPECT_THROW(tokenizer.decode({ 100 }), InvalidParameterError);
}

// ---------------------------------------------------------------------- train

TEST(UnigramTokenizerTrainTest, TrainedTokenizerRoundTripsItsTrainingText) {
    testsupport::SilenceStdout quiet;
    UnigramTokenizer tokenizer(1000);
    const std::string text = "low lower lowest low lower low";

    tokenizer.train(text, 5);

    EXPECT_FALSE(tokenizer.vocab.empty());
    EXPECT_EQ(tokenizer.decode(tokenizer.encode(text)), text);
}

TEST(UnigramTokenizerTrainTest, FrequentWordsBecomeSingleTokens) {
    testsupport::SilenceStdout quiet;
    UnigramTokenizer tokenizer(1000);

    tokenizer.train("the cat cat cat cat cat", 5);

    EXPECT_LE(tokenizer.encode("the cat").size(), 3u);
}

TEST(UnigramTokenizerTrainTest, VocabularyIsPrunedToTheTargetSize) {
    testsupport::SilenceStdout quiet;
    UnigramTokenizer tokenizer(8);

    tokenizer.train("the quick brown fox jumps over lazy dog quick brown fox", 6);

    EXPECT_LE(tokenizer.vocab.size(), 8u);
    EXPECT_EQ(tokenizer.vocab.size(), tokenizer.idToToken.size());
}

TEST(UnigramTokenizerTrainTest, IdsAreContiguousFromZero) {
    testsupport::SilenceStdout quiet;
    UnigramTokenizer tokenizer(1000);

    tokenizer.train("abc abc abd abd", 3);

    for (std::size_t id = 0; id < tokenizer.idToToken.size(); id++) {
        EXPECT_EQ(tokenizer.vocab[tokenizer.idToToken[id]].id, static_cast<int>(id));
    }
}

TEST(UnigramTokenizerTrainTest, InvalidArgumentsThrow) {
    testsupport::SilenceStdout quiet;
    UnigramTokenizer tokenizer(100);

    EXPECT_THROW(tokenizer.train("", 3), InvalidSizeError);
    EXPECT_THROW(tokenizer.train("abc", 0), InvalidParameterError);
    EXPECT_THROW(tokenizer.train("abc", -2), InvalidParameterError);
}
