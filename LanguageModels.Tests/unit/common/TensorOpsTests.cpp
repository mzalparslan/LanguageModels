#include "pch.h"
#include "TensorOps.h"
#include "TestSupport.h"
#include <cmath>
#include <limits>

using testsupport::makeMatrix;
using testsupport::tensorsNear;

// ---------------------------------------------------------------- MatMul2D

TEST(MatMul2DTest, MultipliesKnownMatrices) {
    // [1 2 3]   [ 7  8]   [ 58  64]
    // [4 5 6] x [ 9 10] = [139 154]
    //           [11 12]
    auto A = makeMatrix(2, 3, { 1, 2, 3, 4, 5, 6 });
    auto B = makeMatrix(3, 2, { 7, 8, 9, 10, 11, 12 });
    Tensor<double> C;

    MatMul2D(A, B, C);

    EXPECT_TRUE(tensorsNear(C, makeMatrix(2, 2, { 58, 64, 139, 154 }), 1e-12));
}

TEST(MatMul2DTest, IdentityLeavesMatrixUnchanged) {
    auto A = makeMatrix(2, 2, { 3, -1, 0.5, 4 });
    auto identity = makeMatrix(2, 2, { 1, 0, 0, 1 });
    Tensor<double> C;

    MatMul2D(A, identity, C);

    EXPECT_TRUE(tensorsNear(C, A, 1e-12));
}

TEST(MatMul2DTest, ZeroEntriesAreSkippedWithoutChangingResult) {
    // The sparse shortcut must not alter the product.
    auto A = makeMatrix(2, 3, { 0, 2, 0, 0, 0, 3 });
    auto B = makeMatrix(3, 2, { 1, 2, 3, 4, 5, 6 });
    Tensor<double> C;

    MatMul2D(A, B, C);

    EXPECT_TRUE(tensorsNear(C, makeMatrix(2, 2, { 6, 8, 15, 18 }), 1e-12));
}

TEST(MatMul2DTest, ReusesResultBufferOfMatchingShapeAndClearsIt) {
    auto A = makeMatrix(1, 2, { 1, 1 });
    auto B = makeMatrix(2, 1, { 2, 3 });
    Tensor<double> C({ 1, 1 }, 100.0);

    MatMul2D(A, B, C);

    // A stale value must not leak into the accumulated product.
    EXPECT_DOUBLE_EQ(C[0], 5.0);
}

TEST(MatMul2DTest, ReallocatesResultBufferOfWrongShape) {
    auto A = makeMatrix(2, 2, { 1, 2, 3, 4 });
    auto B = makeMatrix(2, 2, { 1, 0, 0, 1 });
    Tensor<double> C({ 5 }, 9.0);

    MatMul2D(A, B, C);

    EXPECT_EQ(C.shape, (std::vector<std::size_t>{ 2, 2 }));
}

TEST(MatMul2DTest, InnerDimensionMismatchThrowsInvalidSizeError) {
    Tensor<double> A({ 2, 3 });
    Tensor<double> B({ 2, 2 });
    Tensor<double> C;

    EXPECT_THROW(MatMul2D(A, B, C), InvalidSizeError);
}

TEST(MatMul2DTest, NonMatrixOperandsThrowInvalidSizeError) {
    Tensor<double> vector({ 3 });
    Tensor<double> matrix({ 3, 3 });
    Tensor<double> empty({ 0, 3 });
    Tensor<double> C;

    EXPECT_THROW(MatMul2D(vector, matrix, C), InvalidSizeError);
    EXPECT_THROW(MatMul2D(matrix, vector, C), InvalidSizeError);
    EXPECT_THROW(MatMul2D(empty, matrix, C), InvalidSizeError);
}

// --------------------------------------------------------------- transpose2D

TEST(Transpose2DTest, SwapsRowsAndColumns) {
    auto A = makeMatrix(2, 3, { 1, 2, 3, 4, 5, 6 });
    Tensor<double> At;

    transpose2D(A, At);

    EXPECT_TRUE(tensorsNear(At, makeMatrix(3, 2, { 1, 4, 2, 5, 3, 6 }), 1e-12));
}

TEST(Transpose2DTest, TransposingTwiceRestoresOriginal) {
    auto A = makeMatrix(3, 2, { 1, 2, 3, 4, 5, 6 });
    Tensor<double> At, restored;

    transpose2D(A, At);
    transpose2D(At, restored);

    EXPECT_TRUE(tensorsNear(restored, A, 1e-12));
}

TEST(Transpose2DTest, ReusesResultBufferOfMatchingShape) {
    auto A = makeMatrix(2, 3, { 1, 2, 3, 4, 5, 6 });
    Tensor<double> At({ 3, 2 }, -1.0);

    transpose2D(A, At);

    EXPECT_TRUE(tensorsNear(At, makeMatrix(3, 2, { 1, 4, 2, 5, 3, 6 }), 1e-12));
}

TEST(Transpose2DTest, NonMatrixThrowsInvalidSizeError) {
    Tensor<double> vector({ 4 });
    Tensor<double> At;

    EXPECT_THROW(transpose2D(vector, At), InvalidSizeError);
}

// ---------------------------------------------------------------- softmaxRow

TEST(SoftmaxRowTest, EveryRowSumsToOneAndIsPositive) {
    auto mat = makeMatrix(2, 3, { 1, 2, 3, -1, 0, 5 });

    softmaxRow(mat);

    for (std::size_t row = 0; row < 2; row++) {
        double sum = 0.0;
        for (std::size_t col = 0; col < 3; col++) {
            EXPECT_GT(mat[row * 3 + col], 0.0);
            sum += mat[row * 3 + col];
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST(SoftmaxRowTest, MatchesKnownValues) {
    auto mat = makeMatrix(1, 3, { 0, 0, std::log(2.0) });

    softmaxRow(mat);

    // weights 1 : 1 : 2
    EXPECT_NEAR(mat[0], 0.25, 1e-12);
    EXPECT_NEAR(mat[1], 0.25, 1e-12);
    EXPECT_NEAR(mat[2], 0.50, 1e-12);
}

TEST(SoftmaxRowTest, EqualScoresGiveUniformDistribution) {
    auto mat = makeMatrix(1, 4, { 3, 3, 3, 3 });

    softmaxRow(mat);

    for (std::size_t i = 0; i < 4; i++) {
        EXPECT_NEAR(mat[i], 0.25, 1e-12);
    }
}

TEST(SoftmaxRowTest, IsInvariantToAddingAConstantToARow) {
    auto plain = makeMatrix(1, 3, { 1, 2, 3 });
    auto shifted = makeMatrix(1, 3, { 501, 502, 503 });

    softmaxRow(plain);
    softmaxRow(shifted);

    EXPECT_TRUE(tensorsNear(shifted, plain, 1e-12));
}

TEST(SoftmaxRowTest, LargeScoresDoNotOverflow) {
    auto mat = makeMatrix(1, 2, { 1000, 1000 });

    EXPECT_NO_THROW(softmaxRow(mat));
    EXPECT_NEAR(mat[0], 0.5, 1e-12);
}

TEST(SoftmaxRowTest, MaskedScoresGetZeroProbability) {
    // -1e9 is the value attention uses for masked positions.
    auto mat = makeMatrix(1, 3, { 1, -1e9, 2 });

    softmaxRow(mat);

    EXPECT_DOUBLE_EQ(mat[1], 0.0);
    EXPECT_NEAR(mat[0] + mat[2], 1.0, 1e-12);
}

TEST(SoftmaxRowTest, NaNScoreThrowsNonFiniteError) {
    auto mat = makeMatrix(1, 2, { 1, std::numeric_limits<double>::quiet_NaN() });

    EXPECT_THROW(softmaxRow(mat), NonFiniteError);
}

TEST(SoftmaxRowTest, InfiniteScoreThrowsNonFiniteError) {
    auto mat = makeMatrix(1, 2, { 1, std::numeric_limits<double>::infinity() });

    EXPECT_THROW(softmaxRow(mat), NonFiniteError);
}

TEST(SoftmaxRowTest, NonMatrixThrowsInvalidSizeError) {
    Tensor<double> vector({ 3 });

    EXPECT_THROW(softmaxRow(vector), InvalidSizeError);
}
