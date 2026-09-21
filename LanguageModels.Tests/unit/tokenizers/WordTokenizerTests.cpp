#include "pch.h"
#include "WordTokenizer.h"

namespace {
    WordTokenizer translationTokenizer() {
        return WordTokenizer({ "<UNK>", "<SOS>", "<EOS>" }, "<UNK>");
    }
}

TEST(WordTokenizerTest, SpecialTokensGetTheFirstIdsInTheOrderGiven) {
    WordTokenizer tokenizer = translationTokenizer();

    EXPECT_EQ(tokenizer.size(), 3u);
    EXPECT_EQ(tokenizer.requireId("<UNK>"), 0u);
    EXPECT_EQ(tokenizer.requireId("<SOS>"), 1u);
    EXPECT_EQ(tokenizer.requireId("<EOS>"), 2u);
    EXPECT_TRUE(tokenizer.isSpecial(2));
    EXPECT_FALSE(tokenizer.isSpecial(3));
}

TEST(WordTokenizerTest, FitGivesWordsIdsInOrderOfFirstAppearance) {
    WordTokenizer tokenizer = translationTokenizer();

    tokenizer.fit({ "the cat sat", "the dog sat down" });

    EXPECT_EQ(tokenizer.size(), 3u + 5u);
    EXPECT_EQ(tokenizer.id("the"), 3u);
    EXPECT_EQ(tokenizer.id("cat"), 4u);
    EXPECT_EQ(tokenizer.id("sat"), 5u);
    EXPECT_EQ(tokenizer.id("dog"), 6u);
    EXPECT_EQ(tokenizer.id("down"), 7u);
}

TEST(WordTokenizerTest, UnknownWordsBecomeTheUnknownToken) {
    WordTokenizer tokenizer = translationTokenizer();
    tokenizer.fit({ "hello world" });

    EXPECT_EQ(tokenizer.id("goodbye"), tokenizer.requireId("<UNK>"));
    EXPECT_EQ(tokenizer.encode("hello there world"), (std::vector<std::size_t>{ 3, 0, 4 }));
}

TEST(WordTokenizerTest, DecodeSkipsSpecialTokensUnlessAskedNotTo) {
    WordTokenizer tokenizer = translationTokenizer();
    tokenizer.fit({ "hello world" });
    const std::vector<std::size_t> ids = { 1, 3, 4, 2 }; // <SOS> hello world <EOS>

    EXPECT_EQ(tokenizer.decode(ids), (std::vector<std::string>{ "hello", "world" }));
    EXPECT_EQ(tokenizer.decode(ids, false), (std::vector<std::string>{ "<SOS>", "hello", "world", "<EOS>" }));
}

TEST(WordTokenizerTest, EncodeThenDecodeRoundTrips) {
    WordTokenizer tokenizer = translationTokenizer();
    tokenizer.fit({ "one two three" });

    EXPECT_EQ(tokenizer.decode(tokenizer.encode("three one two")),
        (std::vector<std::string>{ "three", "one", "two" }));
}

TEST(WordTokenizerTest, WordLooksUpTheTextOfAnId) {
    WordTokenizer tokenizer = translationTokenizer();
    tokenizer.fit({ "alpha beta" });

    EXPECT_EQ(tokenizer.word(3), "alpha");
    EXPECT_EQ(tokenizer.word(1), "<SOS>");
    EXPECT_THROW(tokenizer.word(5), InvalidParameterError);
}

TEST(WordTokenizerTest, FitAgainReplacesTheWordsButKeepsTheSpecialTokens) {
    WordTokenizer tokenizer = translationTokenizer();
    tokenizer.fit({ "old words here" });

    tokenizer.fit({ "new text" });

    EXPECT_EQ(tokenizer.size(), 3u + 2u);
    EXPECT_EQ(tokenizer.id("new"), 3u);
    EXPECT_EQ(tokenizer.id("old"), tokenizer.requireId("<UNK>"));
    EXPECT_EQ(tokenizer.requireId("<SOS>"), 1u);
}

TEST(WordTokenizerTest, ASpecialTokenAppearingInTheTextIsNotAddedTwice) {
    WordTokenizer tokenizer = translationTokenizer();

    tokenizer.fit({ "<SOS> hello <SOS>" });

    EXPECT_EQ(tokenizer.size(), 4u);
    EXPECT_EQ(tokenizer.id("<SOS>"), 1u);
}

TEST(WordTokenizerTest, MaxVocabSizeKeepsTheMostFrequentWords) {
    WordTokenizer tokenizer({ "<UNK>" }, "<UNK>");

    // b x3, a x2, c x1, d x1: room for special token plus two words.
    tokenizer.fit({ "a b b b", "c a d" }, 3);

    EXPECT_EQ(tokenizer.size(), 3u);
    // a and b are kept, in order of first appearance.
    EXPECT_EQ(tokenizer.id("a"), 1u);
    EXPECT_EQ(tokenizer.id("b"), 2u);
    EXPECT_EQ(tokenizer.id("c"), 0u);
    EXPECT_EQ(tokenizer.id("d"), 0u);
}

TEST(WordTokenizerTest, MaxVocabSizeBreaksTiesInFavourOfEarlierWords) {
    WordTokenizer tokenizer({ "<UNK>" }, "<UNK>");

    tokenizer.fit({ "p q r s" }, 3);

    EXPECT_EQ(tokenizer.id("p"), 1u);
    EXPECT_EQ(tokenizer.id("q"), 2u);
    EXPECT_EQ(tokenizer.id("r"), 0u);
}

TEST(WordTokenizerTest, ZeroMaxVocabSizeKeepsEveryWord) {
    WordTokenizer tokenizer({ "<UNK>" }, "<UNK>");

    tokenizer.fit({ "a b c d e f g h" }, 0);

    EXPECT_EQ(tokenizer.size(), 9u);
}

TEST(WordTokenizerTest, RejectsInvalidConfiguration) {
    EXPECT_THROW(WordTokenizer({ "<SOS>" }, "<UNK>"), InvalidParameterError);
    EXPECT_THROW(WordTokenizer({ "<UNK>", "<UNK>" }, "<UNK>"), InvalidParameterError);

    WordTokenizer tokenizer = translationTokenizer();
    // Three special tokens cannot fit into a vocabulary of two.
    EXPECT_THROW(tokenizer.fit({ "a" }, 2), InvalidParameterSizeError);
}

TEST(WordTokenizerTest, RequireIdAndDecodeRejectIdsAndTokensThatDoNotExist) {
    WordTokenizer tokenizer = translationTokenizer();
    tokenizer.fit({ "a" });

    EXPECT_THROW(tokenizer.requireId("<MASK>"), InvalidParameterError);
    EXPECT_THROW(tokenizer.decode({ 99 }), InvalidParameterError);
}

TEST(WordTokenizerTest, BertStyleSpecialTokensWork) {
    WordTokenizer tokenizer({ "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]" }, "[UNK]");
    tokenizer.fit({ "hello world" });

    EXPECT_EQ(tokenizer.requireId("[PAD]"), 0u);
    EXPECT_EQ(tokenizer.requireId("[MASK]"), 4u);
    EXPECT_EQ(tokenizer.id("hello"), 5u);
    EXPECT_EQ(tokenizer.id("nope"), 1u);
}
