#include "pch.h"
#include "RotaryEmbedding.h"
#include "TestSupport.h"
#include <cmath>
#include <type_traits>

namespace {
    // Two rotation pairs per row: frequencies 1 and 0.01.
    class TinyRopeConfig {
    public:
        static constexpr std::size_t d_head = 4;
        static constexpr std::size_t maxSeqLen = 8;
    };

    using Rope = RotaryEmbedding<double, TinyRopeConfig>;

    // [rows, 4] tensor whose every row is `row`.
    Tensor<double> repeatedRows(std::size_t rows, const std::vector<double>& row) {
        Tensor<double> x({ rows, row.size() });
        for (std::size_t r = 0; r < rows; r++) {
            for (std::size_t j = 0; j < row.size(); j++) {
                x[r * row.size() + j] = row[j];
            }
        }
        return x;
    }

    double dotRows(const Tensor<double>& x, std::size_t a, std::size_t b) {
        double dot = 0.0;
        for (std::size_t j = 0; j < x.shape[1]; j++) {
            dot += x[a * x.shape[1] + j] * x[b * x.shape[1] + j];
        }
        return dot;
    }
}

TEST(RotaryEmbeddingTest, PositionZeroIsLeftUnchanged) {
    Rope rope;
    auto x = repeatedRows(3, { 0.3, -0.7, 1.1, 0.4 });

    rope.apply(x);

    EXPECT_DOUBLE_EQ(x[0], 0.3);
    EXPECT_DOUBLE_EQ(x[1], -0.7);
    EXPECT_DOUBLE_EQ(x[2], 1.1);
    EXPECT_DOUBLE_EQ(x[3], 0.4);
}

TEST(RotaryEmbeddingTest, RotatesEachPairByPositionTimesFrequency) {
    Rope rope;
    auto x = repeatedRows(3, { 1, 0, 1, 0 });

    rope.apply(x);

    // Row 2: first pair rotated by 2 * 1 rad, second pair by 2 * 0.01 rad.
    EXPECT_NEAR(x[2 * 4 + 0], std::cos(2.0), 1e-12);
    EXPECT_NEAR(x[2 * 4 + 1], std::sin(2.0), 1e-12);
    EXPECT_NEAR(x[2 * 4 + 2], std::cos(0.02), 1e-12);
    EXPECT_NEAR(x[2 * 4 + 3], std::sin(0.02), 1e-12);
}

TEST(RotaryEmbeddingTest, RotationPreservesEveryRowNorm) {
    Rope rope;
    auto x = testsupport::patternMatrix(8, 4);
    Tensor<double> original = x;

    rope.apply(x);

    for (std::size_t r = 0; r < 8; r++) {
        EXPECT_NEAR(dotRows(x, r, r), dotRows(original, r, r), 1e-12);
    }
}

TEST(RotaryEmbeddingTest, DotProductDependsOnlyOnRelativePosition) {
    // defining property of RoPE: <R(m) q, R(n) k> depends on m - n only.
    Rope rope;
    auto x = repeatedRows(8, { 0.3, -0.7, 1.1, 0.4 });

    rope.apply(x);

    EXPECT_NEAR(dotRows(x, 1, 0), dotRows(x, 4, 3), 1e-12);
    EXPECT_NEAR(dotRows(x, 4, 3), dotRows(x, 7, 6), 1e-12);
    EXPECT_NEAR(dotRows(x, 5, 2), dotRows(x, 7, 4), 1e-12);
    // ... while a different offset gives a different value.
    EXPECT_GT(std::fabs(dotRows(x, 1, 0) - dotRows(x, 3, 0)), 1e-6);
}

TEST(RotaryEmbeddingTest, DifferentPositionsRotateTheSameVectorDifferently) {
    Rope rope;
    auto x = repeatedRows(4, { 1, 0, 1, 0 });

    rope.apply(x);

    EXPECT_GT(std::fabs(x[1 * 4] - x[2 * 4]), 1e-6);
}

TEST(RotaryEmbeddingTest, ApplyingTwiceMatchesApplyingOnFreshCopies) {
    // cos/sin tables are shared statics; a second instance must agree.
    Rope first;
    Rope second;
    auto a = testsupport::patternMatrix(5, 4);
    auto b = a;

    first.apply(a);
    second.apply(b);

    EXPECT_TRUE(testsupport::tensorsEqual(a, b));
}

TEST(RotaryEmbeddingTest, AcceptsExactlyMaxSeqLenRows) {
    Rope rope;
    auto x = testsupport::patternMatrix(TinyRopeConfig::maxSeqLen, 4);

    EXPECT_NO_THROW(rope.apply(x));
}

TEST(RotaryEmbeddingTest, RejectsSequenceLongerThanMaxSeqLen) {
    Rope rope;
    auto x = testsupport::patternMatrix(TinyRopeConfig::maxSeqLen + 1, 4);

    EXPECT_THROW(rope.apply(x), InvalidSizeError);
}

TEST(RotaryEmbeddingTest, RejectsWrongWidthAndNonMatrixInput) {
    Rope rope;
    auto wrongWidth = testsupport::patternMatrix(3, 6);
    Tensor<double> vector({ 4 }, 1.0);

    EXPECT_THROW(rope.apply(wrongWidth), InvalidSizeError);
    EXPECT_THROW(rope.apply(vector), InvalidSizeError);
}

TEST(RotaryEmbeddingTest, InverseUndoesTheRotation) {
    Rope rope;
    auto x = testsupport::patternMatrix(8, 4);
    Tensor<double> original = x;

    rope.apply(x);
    rope.applyInverse(x);

    EXPECT_TRUE(testsupport::tensorsNear(x, original, 1e-12));
}

TEST(RotaryEmbeddingTest, InverseRotatesByTheOppositeAngle) {
    Rope rope;
    auto x = repeatedRows(3, { 1, 0, 1, 0 });

    rope.applyInverse(x);

    // Row 2: first pair rotated by -2 rad, second pair by -0.02 rad.
    EXPECT_NEAR(x[2 * 4 + 0], std::cos(2.0), 1e-12);
    EXPECT_NEAR(x[2 * 4 + 1], -std::sin(2.0), 1e-12);
    EXPECT_NEAR(x[2 * 4 + 2], std::cos(0.02), 1e-12);
    EXPECT_NEAR(x[2 * 4 + 3], -std::sin(0.02), 1e-12);
}

TEST(RotaryEmbeddingTest, InverseLeavesPositionZeroUnchangedAndValidatesInput) {
    Rope rope;
    auto x = repeatedRows(2, { 0.3, -0.7, 1.1, 0.4 });

    rope.applyInverse(x);

    EXPECT_DOUBLE_EQ(x[0], 0.3);
    EXPECT_DOUBLE_EQ(x[3], 0.4);
    auto tooLong = testsupport::patternMatrix(TinyRopeConfig::maxSeqLen + 1, 4);
    auto wrongWidth = testsupport::patternMatrix(3, 6);
    EXPECT_THROW(rope.applyInverse(tooLong), InvalidSizeError);
    EXPECT_THROW(rope.applyInverse(wrongWidth), InvalidSizeError);
}

TEST(RotaryEmbeddingTest, DefaultConfigIsModelConfig) {
    // Compile-time check that default template argument stays ModelConfig.
    static_assert(std::is_same_v<RotaryEmbedding<double>, RotaryEmbedding<double, ModelConfig>>);
    SUCCEED();
}
