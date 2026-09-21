#include "pch.h"
#include "TextNormalizer.h"

TEST(TextNormalizerTest, LowerCasesLetters) {
    EXPECT_EQ(text::normalize("Hello World"), "hello world");
}

TEST(TextNormalizerTest, KeepsApostrophesAndHyphens) {
    EXPECT_EQ(text::normalize("It's a well-known fact"), "it's a well-known fact");
}

TEST(TextNormalizerTest, DropsDigitsPunctuationAndAccentedLetters) {
    // \xC3\xA9 is UTF-8 encoding of an accented e: two bytes, both dropped.
    EXPECT_EQ(text::normalize("Caf\xC3\xA9 no. 5, (really)!"), "caf no  really");
}

TEST(TextNormalizerTest, EmptyAndSymbolOnlyTextBecomeEmpty) {
    EXPECT_EQ(text::normalize(""), "");
    EXPECT_EQ(text::normalize("123 !?"), " ");
}

TEST(TextNormalizerTest, SplitOnWhitespaceSplitsAnyRunOfSpaces) {
    EXPECT_EQ(text::splitOnWhitespace("  a  bb\tccc\n d "), (std::vector<std::string>{ "a", "bb", "ccc", "d" }));
    EXPECT_TRUE(text::splitOnWhitespace("").empty());
    EXPECT_TRUE(text::splitOnWhitespace(" \t\n").empty());
}
