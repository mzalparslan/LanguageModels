#include "pch.h"
#include "SinusoidalEmbedding.h"
#include "TestSupport.h"
#include <cmath>

TEST(SinusoidalEmbeddingTest, TableHasMaxLenByWidthShape) {
    SinusoidalEmbedding<double> embedding(10, 4);

    EXPECT_EQ(embedding.pe.shape, (std::vector<std::size_t>{ 10, 4 }));
}

TEST(SinusoidalEmbeddingTest, PositionZeroIsSineZeroAndCosineOne) {
    SinusoidalEmbedding<double> embedding(10, 4);

    EXPECT_DOUBLE_EQ(embedding.pe[0], 0.0);
    EXPECT_DOUBLE_EQ(embedding.pe[1], 1.0);
    EXPECT_DOUBLE_EQ(embedding.pe[2], 0.0);
    EXPECT_DOUBLE_EQ(embedding.pe[3], 1.0);
}

TEST(SinusoidalEmbeddingTest, PositionOneUsesGeometricallySpacedFrequencies) {
    const std::size_t dim = 4;
    SinusoidalEmbedding<double> embedding(10, dim);

    // Pair i has frequency 10000^(-i / dim): 1 for the first pair, 0.01 for the second.
    EXPECT_NEAR(embedding.pe[1 * dim + 0], std::sin(1.0), 1e-12);
    EXPECT_NEAR(embedding.pe[1 * dim + 1], std::cos(1.0), 1e-12);
    EXPECT_NEAR(embedding.pe[1 * dim + 2], std::sin(0.01), 1e-12);
    EXPECT_NEAR(embedding.pe[1 * dim + 3], std::cos(0.01), 1e-12);
}

TEST(SinusoidalEmbeddingTest, EveryValueIsBoundedByOne) {
    SinusoidalEmbedding<double> embedding(64, 16);

    for (std::size_t i = 0; i < embedding.pe.size(); i++) {
        EXPECT_LE(std::fabs(embedding.pe[i]), 1.0 + 1e-12);
    }
}

TEST(SinusoidalEmbeddingTest, DistinctPositionsGetDistinctEncodings) {
    const std::size_t dim = 8;
    SinusoidalEmbedding<double> embedding(20, dim);

    for (std::size_t a = 0; a < 20; a++) {
        for (std::size_t b = a + 1; b < 20; b++) {
            bool differs = false;
            for (std::size_t j = 0; j < dim; j++) {
                differs = differs || embedding.pe[a * dim + j] != embedding.pe[b * dim + j];
            }
            EXPECT_TRUE(differs) << "positions " << a << " and " << b << " are identical";
        }
    }
}

TEST(SinusoidalEmbeddingTest, OddWidthLeavesTheLastCosineSlotOut) {
    SinusoidalEmbedding<double> embedding(4, 3);

    EXPECT_EQ(embedding.pe.shape, (std::vector<std::size_t>{ 4, 3 }));
    // Position 1, last column: sine of the third-pair frequency.
    EXPECT_NEAR(embedding.pe[1 * 3 + 2], std::sin(std::exp(2 * -std::log(10000.0) / 3)), 1e-12);
}

TEST(SinusoidalEmbeddingTest, ForwardAddsPositionEncodingToInput) {
    SinusoidalEmbedding<double> embedding(10, 4);
    Tensor<double> x({ 3, 4 }, 1.0);

    embedding.forward(x);

    for (std::size_t i = 0; i < x.size(); i++) {
        EXPECT_NEAR(x[i], 1.0 + embedding.pe[i], 1e-12);
    }
}

TEST(SinusoidalEmbeddingTest, ForwardOnZerosReproducesTheTable) {
    SinusoidalEmbedding<double> embedding(10, 4);
    Tensor<double> x({ 5, 4 }, 0.0);

    embedding.forward(x);

    for (std::size_t i = 0; i < x.size(); i++) {
        EXPECT_DOUBLE_EQ(x[i], embedding.pe[i]);
    }
}

TEST(SinusoidalEmbeddingTest, ForwardAcceptsExactlyMaxLenRows) {
    SinusoidalEmbedding<double> embedding(6, 4);
    Tensor<double> x({ 6, 4 }, 0.0);

    EXPECT_NO_THROW(embedding.forward(x));
}

TEST(SinusoidalEmbeddingTest, ForwardRejectsSequenceLongerThanMaxLen) {
    SinusoidalEmbedding<double> embedding(6, 4);
    Tensor<double> x({ 7, 4 }, 0.0);

    EXPECT_THROW(embedding.forward(x), InvalidSizeError);
}

TEST(SinusoidalEmbeddingTest, ForwardRejectsWrongWidthAndNonMatrix) {
    SinusoidalEmbedding<double> embedding(6, 4);
    Tensor<double> wrongWidth({ 3, 5 }, 0.0);
    Tensor<double> vector({ 4 }, 0.0);

    EXPECT_THROW(embedding.forward(wrongWidth), InvalidSizeError);
    EXPECT_THROW(embedding.forward(vector), InvalidSizeError);
}

TEST(SinusoidalEmbeddingTest, ZeroSizesThrowInvalidParameterSizeError) {
    EXPECT_THROW(SinusoidalEmbedding<double>(0, 4), InvalidParameterSizeError);
    EXPECT_THROW(SinusoidalEmbedding<double>(6, 0), InvalidParameterSizeError);
}

TEST(SinusoidalEmbeddingTest, HasNoLearnableParameters) {
    SinusoidalEmbedding<double> embedding(6, 4);
    Tensor<double> before = embedding.pe;

    embedding.zeroGrad();
    embedding.update(0.1);
    embedding.backward({ 0, 1 }, Tensor<double>({ 2, 4 }, 1.0));

    EXPECT_TRUE(testsupport::tensorsEqual(embedding.pe, before));
}
