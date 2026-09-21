#include "pch.h"
#include "DataLoader.h"
#include <atomic>
#include <filesystem>
#include <fstream>

namespace {
    // A file in temp folder that exists for one test.
    class TempFile {
    public:
        explicit TempFile(const std::string& contents) {
            static std::atomic<int> counter{ 0 };
            path = std::filesystem::temp_directory_path()
                / ("lm_dataloader_test_" + std::to_string(counter++) + ".txt");
            std::ofstream file(path, std::ios::binary);
            file << contents;
        }
        ~TempFile() {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
        TempFile(const TempFile&) = delete;
        TempFile& operator=(const TempFile&) = delete;

        std::filesystem::path path;
    };

    const std::filesystem::path missingFile = std::filesystem::temp_directory_path() / "lm_no_such_file.txt";
}

// ----------------------------------------------------------- sentence pairs

TEST(DataLoaderTest, LoadsTabSeparatedSentencePairsAndIgnoresTheAttributionColumn) {
    TempFile file("Hi.\tSalut !\tCC-BY 2.0 (France) Attribution: tatoeba.org #1\n"
        "Run away!\tCours !\tCC-BY 2.0\n");

    auto pairs = DataLoader::loadSentencePairs(file.path);

    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].source, "hi");
    EXPECT_EQ(pairs[0].target, "salut ");
    EXPECT_EQ(pairs[1].source, "run away");
    EXPECT_EQ(pairs[1].target, "cours ");
}

TEST(DataLoaderTest, SentencePairsAreNormalizedLikeText) {
    // Accents and digits are dropped; apostrophes and hyphens stay.
    TempFile file("It's 5 o'clock\tC'est l'\xC3\xA9t\xC3\xA9 - fini\n");

    auto pairs = DataLoader::loadSentencePairs(file.path);

    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].source, "it's  o'clock");
    EXPECT_EQ(pairs[0].target, "c'est l't - fini");
}

TEST(DataLoaderTest, LinesWithoutATabOrWithAnEmptySideAreSkipped) {
    TempFile file("no tab here\n"
        "ok\tbon\n"
        "\tonly target\n"
        "only source\t\n"
        "123\t456\n"          // both sides are empty after normalizing
        "\n"
        "yes\toui\n");

    auto pairs = DataLoader::loadSentencePairs(file.path);

    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].source, "ok");
    EXPECT_EQ(pairs[1].source, "yes");
}

TEST(DataLoaderTest, SentencePairLimitStopsReading) {
    TempFile file("a\tb\nc\td\ne\tf\ng\th\n");

    EXPECT_EQ(DataLoader::loadSentencePairs(file.path, 2).size(), 2u);
    EXPECT_EQ(DataLoader::loadSentencePairs(file.path, 0).size(), 4u);
    EXPECT_EQ(DataLoader::loadSentencePairs(file.path, 100).size(), 4u);
}

TEST(DataLoaderTest, WindowsLineEndingsGiveTheSamePairs) {
    TempFile lf("hello\tbonjour\nbye\tsalut\n");
    TempFile crlf("hello\tbonjour\r\nbye\tsalut\r\n");

    auto fromLf = DataLoader::loadSentencePairs(lf.path);
    auto fromCrlf = DataLoader::loadSentencePairs(crlf.path);

    ASSERT_EQ(fromLf.size(), fromCrlf.size());
    for (std::size_t i = 0; i < fromLf.size(); i++) {
        EXPECT_EQ(fromLf[i].source, fromCrlf[i].source);
        EXPECT_EQ(fromLf[i].target, fromCrlf[i].target);
    }
}

TEST(DataLoaderTest, SentencePairsThrowDataLoadErrorForAMissingFileOrNoPairs) {
    EXPECT_THROW((void)DataLoader::loadSentencePairs(missingFile), DataLoadError);

    TempFile empty("");
    TempFile noPairs("just a line\nanother line\n");
    EXPECT_THROW((void)DataLoader::loadSentencePairs(empty.path), DataLoadError);
    EXPECT_THROW((void)DataLoader::loadSentencePairs(noPairs.path), DataLoadError);
}

// --------------------------------------------------------------------- text

TEST(DataLoaderTest, LoadTextReturnsTheWholeFileWithoutCarriageReturns) {
    TempFile file("First line.\r\nSecond line.\r\n");

    EXPECT_EQ(DataLoader::loadText(file.path), "First line.\nSecond line.\n");
}

TEST(DataLoaderTest, LoadCharactersReturnsEveryCharacter) {
    TempFile file("ab\nc");

    EXPECT_EQ(DataLoader::loadCharacters(file.path), (std::vector<char>{ 'a', 'b', '\n', 'c' }));
}

TEST(DataLoaderTest, TextLoadersThrowDataLoadErrorForAMissingOrEmptyFile) {
    TempFile empty("");

    EXPECT_THROW((void)DataLoader::loadText(missingFile), DataLoadError);
    EXPECT_THROW((void)DataLoader::loadText(empty.path), DataLoadError);
    EXPECT_THROW((void)DataLoader::loadCharacters(missingFile), DataLoadError);
    EXPECT_THROW((void)DataLoader::loadCharacters(empty.path), DataLoadError);
}

// -------------------------------------------------------------------- lines

TEST(DataLoaderTest, LoadLinesNormalizesAndSkipsEmptyLines) {
    TempFile file("First Citizen:\n"
        "\n"
        "Before we proceed, hear me speak.\n"
        "   \n"
        "42\n"                // empty after normalizing
        "All: Speak, speak.\n");

    auto lines = DataLoader::loadLines(file.path);

    ASSERT_EQ(lines.size(), 3u);
    EXPECT_EQ(lines[0], "first citizen");
    EXPECT_EQ(lines[1], "before we proceed hear me speak");
    EXPECT_EQ(lines[2], "all speak speak");
}

TEST(DataLoaderTest, LoadLinesHonoursTheLimit) {
    TempFile file("one\ntwo\nthree\nfour\n");

    EXPECT_EQ(DataLoader::loadLines(file.path, 2), (std::vector<std::string>{ "one", "two" }));
}

TEST(DataLoaderTest, LoadLinesThrowsDataLoadErrorForAMissingFileOrNoUsableLine) {
    TempFile blank("\n \n123\n");

    EXPECT_THROW((void)DataLoader::loadLines(missingFile), DataLoadError);
    EXPECT_THROW((void)DataLoader::loadLines(blank.path), DataLoadError);
}

TEST(DataLoaderTest, DataLoadErrorIsARuntimeError) {
    static_assert(std::is_base_of_v<std::runtime_error, DataLoadError>);
    EXPECT_THROW((void)DataLoader::loadText(missingFile), std::runtime_error);
}
