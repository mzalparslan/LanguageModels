#include "pch.h"
#include "CharTokenizer.h"

namespace {
    std::vector<char> chars(const std::string& text) {
        return std::vector<char>(text.begin(), text.end());
    }
}

TEST(CharTokenizerTest, CharactersGetIdsFromOneInAscendingOrder) {
    CharTokenizer tokenizer;

    tokenizer.fit(chars("banana"));

    // Distinct characters: a, b, n.
    EXPECT_EQ(tokenizer.size(), 4u);
    EXPECT_EQ(tokenizer.encode(chars("abn")), (std::vector<std::size_t>{ 1, 2, 3 }));
}

TEST(CharTokenizerTest, EncodeMapsUnseenCharactersToTheUnknownId) {
    CharTokenizer tokenizer;
    tokenizer.fit(chars("abc"));

    EXPECT_EQ(tokenizer.encode(chars("azb")), (std::vector<std::size_t>{ 1, CharTokenizer::unknownId, 2 }));
    EXPECT_EQ(CharTokenizer::unknownId, 0u);
}

TEST(CharTokenizerTest, DecodeRoundTripsKnownCharacters) {
    CharTokenizer tokenizer;
    tokenizer.fit(chars("hello world"));

    EXPECT_EQ(tokenizer.decode(tokenizer.encode(chars("hello"))), "hello");
}

TEST(CharTokenizerTest, DecodeShowsTheUnknownIdAsAQuestionMark) {
    CharTokenizer tokenizer;
    tokenizer.fit(chars("ab"));

    EXPECT_EQ(tokenizer.decode({ 1, 0, 2 }), "a?b");
}

TEST(CharTokenizerTest, DecodeRejectsAnIdOutsideTheVocabulary) {
    CharTokenizer tokenizer;
    tokenizer.fit(chars("ab"));

    EXPECT_THROW(tokenizer.decode({ 3 }), InvalidParameterError);
}

TEST(CharTokenizerTest, FitAgainReplacesTheVocabulary) {
    CharTokenizer tokenizer;
    tokenizer.fit(chars("abc"));

    tokenizer.fit(chars("xyz"));

    EXPECT_EQ(tokenizer.size(), 4u);
    EXPECT_EQ(tokenizer.encode(chars("ax")), (std::vector<std::size_t>{ 0, 1 }));
}

TEST(CharTokenizerTest, NewlinesAndHighBytesAreOrdinaryCharacters) {
    CharTokenizer tokenizer;
    tokenizer.fit(std::vector<char>{ '\n', static_cast<char>(0xC3), 'a' });

    EXPECT_EQ(tokenizer.size(), 4u);
    // Ascending byte order: '\n' (10), 'a' (97), 0xC3 (195).
    EXPECT_EQ(tokenizer.encode(std::vector<char>{ '\n', 'a', static_cast<char>(0xC3) }),
        (std::vector<std::size_t>{ 1, 2, 3 }));
}

TEST(CharTokenizerTest, AnUnfittedTokenizerKnowsOnlyTheUnknownId) {
    CharTokenizer tokenizer;

    EXPECT_EQ(tokenizer.size(), 1u);
    EXPECT_EQ(tokenizer.encode(chars("a")), (std::vector<std::size_t>{ 0 }));
}
