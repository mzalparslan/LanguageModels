#include "pch.h"
#include "WordPieceTokenizer.h"
#include "TestSupport.h"
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {
    // Vocabulary in file order, so the ids are the indices.
    const std::vector<std::string> vocabularyLines = {
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",   // 0..4
        "hello", "world", "play", "##ing", "##s", "un", "##able"   // 5..11
    };

    enum Id : std::size_t {
        Pad = 0, Unk = 1, Cls = 2, Sep = 3, Mask = 4,
        Hello = 5, World = 6, Play = 7, Ing = 8, S = 9, Un = 10, Able = 11
    };

    using Ids = std::vector<std::size_t>;
}

// Writes the vocabulary to a temporary file the tests load, and removes it afterwards.
class WordPieceTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        path = (std::filesystem::temp_directory_path() / "languagemodels_wordpiece_test_vocab.txt").string();
    }

    void TearDown() override {
        std::remove(path.c_str());
    }

    void writeVocabulary(const std::string& lineEnding = "\n") {
        std::ofstream file(path, std::ios::binary);
        for (const auto& line : vocabularyLines) {
            file << line << lineEnding;
        }
    }

    WordPieceTokenizer loadedTokenizer() {
        writeVocabulary();
        testsupport::SilenceStdout quiet;
        WordPieceTokenizer tokenizer;
        tokenizer.loadVocab(path);
        return tokenizer;
    }

    std::string path;
};

// ------------------------------------------------------------------ loadVocab

TEST_F(WordPieceTokenizerTest, LoadVocabAssignsIdsInFileOrder) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.vocab.size(), vocabularyLines.size());
    EXPECT_EQ(tokenizer.idToWord, vocabularyLines);
    EXPECT_EQ(tokenizer.vocab["hello"], static_cast<std::size_t>(Hello));
    EXPECT_EQ(tokenizer.vocab["##ing"], static_cast<std::size_t>(Ing));
}

TEST_F(WordPieceTokenizerTest, LoadVocabRecordsTheSpecialTokenIds) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.padId, static_cast<std::size_t>(Pad));
    EXPECT_EQ(tokenizer.unkId, static_cast<std::size_t>(Unk));
    EXPECT_EQ(tokenizer.clsId, static_cast<std::size_t>(Cls));
    EXPECT_EQ(tokenizer.sepId, static_cast<std::size_t>(Sep));
    EXPECT_EQ(tokenizer.maskId, static_cast<std::size_t>(Mask));
}

TEST_F(WordPieceTokenizerTest, LoadVocabAcceptsWindowsLineEndings) {
    writeVocabulary("\r\n");
    testsupport::SilenceStdout quiet;
    WordPieceTokenizer tokenizer;

    tokenizer.loadVocab(path);

    EXPECT_EQ(tokenizer.vocab.count("hello"), 1u);
    EXPECT_EQ(tokenizer.vocab.count("hello\r"), 0u);
    EXPECT_EQ(tokenizer.clsId, static_cast<std::size_t>(Cls));
}

TEST_F(WordPieceTokenizerTest, LoadingAgainReplacesThePreviousVocabulary) {
    auto tokenizer = loadedTokenizer();
    {
        std::ofstream file(path, std::ios::binary);
        file << "[PAD]\n[UNK]\nonly\n";
    }
    testsupport::SilenceStdout quiet;

    tokenizer.loadVocab(path);

    EXPECT_EQ(tokenizer.vocab.size(), 3u);
    EXPECT_EQ(tokenizer.vocab.count("hello"), 0u);
}

TEST_F(WordPieceTokenizerTest, MissingVocabularyFileLeavesTheTokenizerUnchanged) {
    auto tokenizer = loadedTokenizer();
    testsupport::SilenceStderr quiet;

    EXPECT_NO_THROW(tokenizer.loadVocab(path + ".does-not-exist"));

    EXPECT_EQ(tokenizer.vocab.size(), vocabularyLines.size());
}

TEST(WordPieceTokenizerDefaultTest, StartsEmptyWithAllSpecialIdsZero) {
    WordPieceTokenizer tokenizer;

    EXPECT_TRUE(tokenizer.vocab.empty());
    EXPECT_EQ(tokenizer.unkId, 0u);
    EXPECT_EQ(tokenizer.clsId, 0u);
}

// ------------------------------------------------------------- basicTokenize

TEST(WordPieceBasicTokenizeTest, SplitsOnAnyWhitespaceAndSkipsEmptyWords) {
    WordPieceTokenizer tokenizer;

    auto words = tokenizer.basicTokenize("  hello   world \t\n play ");

    EXPECT_EQ(words, (std::vector<std::string>{ "hello", "world", "play" }));
}

TEST(WordPieceBasicTokenizeTest, EmptyOrBlankTextGivesNoWords) {
    WordPieceTokenizer tokenizer;

    EXPECT_TRUE(tokenizer.basicTokenize("").empty());
    EXPECT_TRUE(tokenizer.basicTokenize("  \t\n ").empty());
}

TEST(WordPieceBasicTokenizeTest, KeepsPunctuationInsideWords) {
    WordPieceTokenizer tokenizer;

    EXPECT_EQ(tokenizer.basicTokenize("hello, world!"), (std::vector<std::string>{ "hello,", "world!" }));
}

// ---------------------------------------------------------- wordPieceTokenize

TEST_F(WordPieceTokenizerTest, WholeWordInVocabularyStaysWhole) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.wordPieceTokenize("hello"), (std::vector<std::string>{ "hello" }));
}

TEST_F(WordPieceTokenizerTest, SplitsIntoLongestKnownPiecesWithContinuationPrefix) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.wordPieceTokenize("playing"), (std::vector<std::string>{ "play", "##ing" }));
    EXPECT_EQ(tokenizer.wordPieceTokenize("plays"), (std::vector<std::string>{ "play", "##s" }));
    EXPECT_EQ(tokenizer.wordPieceTokenize("unable"), (std::vector<std::string>{ "un", "##able" }));
}

TEST_F(WordPieceTokenizerTest, WordThatCannotBeCoveredBecomesUnknown) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.wordPieceTokenize("xyz"), (std::vector<std::string>{ "[UNK]" }));
    // One unknown piece makes the whole word unknown.
    EXPECT_EQ(tokenizer.wordPieceTokenize("playxyz"), (std::vector<std::string>{ "[UNK]" }));
}

TEST_F(WordPieceTokenizerTest, OverlongWordsAreUnknown) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.wordPieceTokenize(std::string(101, 'a')), (std::vector<std::string>{ "[UNK]" }));
}

// --------------------------------------------------------------------- encode

TEST_F(WordPieceTokenizerTest, EncodeMapsWordsAndPiecesToIds) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.encode("hello world"), (Ids{ Hello, World }));
    EXPECT_EQ(tokenizer.encode("playing hello"), (Ids{ Play, Ing, Hello }));
}

TEST_F(WordPieceTokenizerTest, EncodeUsesTheUnknownIdForUnknownWords) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.encode("hello xyz world"), (Ids{ Hello, Unk, World }));
}

TEST_F(WordPieceTokenizerTest, EncodeOfEmptyTextGivesNoIds) {
    auto tokenizer = loadedTokenizer();

    EXPECT_TRUE(tokenizer.encode("").empty());
    EXPECT_TRUE(tokenizer.encode("   ").empty());
}

TEST(WordPieceEncodeTest, EncodeBeforeLoadingAVocabularyThrowsInvalidSizeError) {
    WordPieceTokenizer tokenizer;

    EXPECT_THROW(tokenizer.encode("hello"), InvalidSizeError);
}

// ------------------------------------------------------------ encodeSequence

TEST_F(WordPieceTokenizerTest, SingleSentenceIsWrappedInClsAndSep) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.encodeSequence("hello world"), (Ids{ Cls, Hello, World, Sep }));
}

TEST_F(WordPieceTokenizerTest, SentencePairHasASeparatorAfterEachSentence) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.encodeSequence("hello", "playing world"),
        (Ids{ Cls, Hello, Sep, Play, Ing, World, Sep }));
}

TEST_F(WordPieceTokenizerTest, EmptySecondSentenceIsOmitted) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.encodeSequence("hello", ""), (Ids{ Cls, Hello, Sep }));
}

TEST_F(WordPieceTokenizerTest, EmptyFirstSentenceStillGivesClsAndSep) {
    auto tokenizer = loadedTokenizer();

    EXPECT_EQ(tokenizer.encodeSequence(""), (Ids{ Cls, Sep }));
}
