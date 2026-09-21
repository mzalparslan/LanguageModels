#include "pch.h"
#include "DataSplitter.h"
#include "SentencePair.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace {
    std::vector<int> numbers(int count) {
        std::vector<int> values(count);
        std::iota(values.begin(), values.end(), 0);
        return values;
    }
}

// ---------------------------------------------------------------- shuffled

TEST(DataSplitterTest, TrainTestSplitFollowsTheRatio) {
    auto dataset = DataSplitter::trainTestSplit(numbers(10), 0.8);

    EXPECT_EQ(dataset.trainingData.size(), 8u);
    EXPECT_EQ(dataset.testData.size(), 2u);
}

TEST(DataSplitterTest, TrainTestSplitDefaultsToEightyTwenty) {
    auto dataset = DataSplitter::trainTestSplit(numbers(100));

    EXPECT_EQ(dataset.trainingData.size(), 80u);
    EXPECT_EQ(dataset.testData.size(), 20u);
}

TEST(DataSplitterTest, TrainTestSplitKeepsEverySampleExactlyOnce) {
    auto dataset = DataSplitter::trainTestSplit(numbers(50), 0.6);

    std::vector<int> all = dataset.trainingData;
    all.insert(all.end(), dataset.testData.begin(), dataset.testData.end());
    std::sort(all.begin(), all.end());
    EXPECT_EQ(all, numbers(50));
}

TEST(DataSplitterTest, TrainTestSplitShufflesButIsReproducibleForASeed) {
    auto first = DataSplitter::trainTestSplit(numbers(100), 0.8, 7);
    auto again = DataSplitter::trainTestSplit(numbers(100), 0.8, 7);
    auto other = DataSplitter::trainTestSplit(numbers(100), 0.8, 8);

    EXPECT_EQ(first.trainingData, again.trainingData);
    EXPECT_EQ(first.testData, again.testData);
    EXPECT_NE(first.trainingData, other.trainingData);
    // Not just cut in place: training part is not first 80 in order.
    const std::vector<int> inOrder = numbers(100);
    EXPECT_NE(first.trainingData, std::vector<int>(inOrder.begin(), inOrder.begin() + 80));
}

TEST(DataSplitterTest, TrainTestSplitWorksForSentencePairs) {
    std::vector<SentencePair> pairs = { {"a", "b"}, {"c", "d"}, {"e", "f"}, {"g", "h"}, {"i", "j"} };

    auto dataset = DataSplitter::trainTestSplit(pairs, 0.6);

    EXPECT_EQ(dataset.trainingData.size(), 3u);
    EXPECT_EQ(dataset.testData.size(), 2u);
    for (const SentencePair& pair : dataset.trainingData) {
        // Pairs stay together.
        EXPECT_EQ(static_cast<char>(pair.source[0] + 1), pair.target[0]);
    }
}

// -------------------------------------------------------------- sequential

TEST(DataSplitterTest, SequentialSplitKeepsTheOrderAndCutsOnce) {
    auto dataset = DataSplitter::sequentialSplit(numbers(10), 0.7);

    EXPECT_EQ(dataset.trainingData, (std::vector<int>{ 0, 1, 2, 3, 4, 5, 6 }));
    EXPECT_EQ(dataset.testData, (std::vector<int>{ 7, 8, 9 }));
}

TEST(DataSplitterTest, SequentialSplitDefaultsToNinetyTen) {
    auto dataset = DataSplitter::sequentialSplit(numbers(100));

    EXPECT_EQ(dataset.trainingData.size(), 90u);
    EXPECT_EQ(dataset.testData.size(), 10u);
}

TEST(DataSplitterTest, SequentialSplitWorksForCharactersAndSentences) {
    std::vector<char> text = { 'a', 'b', 'c', 'd' };
    std::vector<std::string> lines = { "one", "two", "three", "four" };

    auto characters = DataSplitter::sequentialSplit(text, 0.5);
    auto sentences = DataSplitter::sequentialSplit(lines, 0.75);

    EXPECT_EQ(characters.trainingData, (std::vector<char>{ 'a', 'b' }));
    EXPECT_EQ(characters.testData, (std::vector<char>{ 'c', 'd' }));
    EXPECT_EQ(sentences.trainingData, (std::vector<std::string>{ "one", "two", "three" }));
    EXPECT_EQ(sentences.testData, (std::vector<std::string>{ "four" }));
}

// ------------------------------------------------------------------ errors

TEST(DataSplitterTest, RejectsRatiosOutsideZeroToOne) {
    for (double ratio : { 0.0, 1.0, -0.5, 1.5 }) {
        EXPECT_THROW((void)DataSplitter::trainTestSplit(numbers(10), ratio), InvalidParameterError) << ratio;
        EXPECT_THROW((void)DataSplitter::sequentialSplit(numbers(10), ratio), InvalidParameterError) << ratio;
    }
}

TEST(DataSplitterTest, RejectsRatiosThatAreNotFinite) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    EXPECT_THROW((void)DataSplitter::trainTestSplit(numbers(10), nan), InvalidParameterError);
    EXPECT_THROW((void)DataSplitter::trainTestSplit(numbers(10), inf), InvalidParameterError);
    EXPECT_THROW((void)DataSplitter::sequentialSplit(numbers(10), nan), InvalidParameterError);
}

TEST(DataSplitterTest, RejectsEmptyOrTooSmallSampleSets) {
    EXPECT_THROW((void)DataSplitter::trainTestSplit(std::vector<int>{}), InvalidSizeError);
    EXPECT_THROW((void)DataSplitter::sequentialSplit(std::vector<int>{}), InvalidSizeError);
    // One sample cannot fill both parts.
    EXPECT_THROW((void)DataSplitter::trainTestSplit(numbers(1)), InvalidSizeError);
    EXPECT_THROW((void)DataSplitter::sequentialSplit(numbers(1)), InvalidSizeError);
    // 0.1 of two samples rounds down to no training sample at all.
    EXPECT_THROW((void)DataSplitter::trainTestSplit(numbers(2), 0.1), InvalidSizeError);
    // 0.99 of ten rounds down to nine, leaving one: still fine.
    EXPECT_NO_THROW((void)DataSplitter::trainTestSplit(numbers(10), 0.99));
}
