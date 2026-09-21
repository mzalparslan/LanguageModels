#include "pch.h"
#include "Embedding.h"
#include "TestSupport.h"

using testsupport::makeMatrix;
using testsupport::tensorsNear;

TEST(EmbeddingTest, ConstructionCreatesVocabByWidthTable) {
    RandomEngine rng(42);

    Embedding<double> embedding(10, 4, rng);

    EXPECT_EQ(embedding.table.value.shape, (std::vector<std::size_t>{ 10, 4 }));
    EXPECT_EQ(embedding.vocabSize, 10u);
    EXPECT_EQ(embedding.dModel, 4u);
}

TEST(EmbeddingTest, ZeroSizesThrowInvalidParameterSizeError) {
    RandomEngine rng(42);

    EXPECT_THROW(Embedding<double>(0, 4, rng), InvalidParameterSizeError);
    EXPECT_THROW(Embedding<double>(10, 0, rng), InvalidParameterSizeError);
}

TEST(EmbeddingTest, ForwardLooksUpTableRowsInOrder) {
    RandomEngine rng(42);
    Embedding<double> embedding(5, 3, rng);
    Tensor<double> out;

    embedding.forward({ 2, 0, 2 }, out);

    ASSERT_EQ(out.shape, (std::vector<std::size_t>{ 3, 3 }));
    for (std::size_t j = 0; j < 3; j++) {
        EXPECT_DOUBLE_EQ(out[0 * 3 + j], embedding.table.value[2 * 3 + j]);
        EXPECT_DOUBLE_EQ(out[1 * 3 + j], embedding.table.value[0 * 3 + j]);
        EXPECT_DOUBLE_EQ(out[2 * 3 + j], embedding.table.value[2 * 3 + j]);
    }
}

TEST(EmbeddingTest, ForwardWithEmptySequenceThrowsInvalidSizeError) {
    RandomEngine rng(42);
    Embedding<double> embedding(5, 3, rng);
    Tensor<double> out;

    EXPECT_THROW(embedding.forward({}, out), InvalidSizeError);
}

TEST(EmbeddingTest, ForwardWithTokenAtOrBeyondVocabThrowsInvalidParameterError) {
    RandomEngine rng(42);
    Embedding<double> embedding(5, 3, rng);
    Tensor<double> out;

    EXPECT_NO_THROW(embedding.forward({ 4 }, out));
    EXPECT_THROW(embedding.forward({ 5 }, out), InvalidParameterError);
    EXPECT_THROW(embedding.forward({ 1, 99 }, out), InvalidParameterError);
}

TEST(EmbeddingTest, BackwardAccumulatesIntoTheRowsThatWereUsed) {
    RandomEngine rng(42);
    Embedding<double> embedding(4, 2, rng);
    auto dOut = makeMatrix(3, 2, { 1, 2, 10, 20, 100, 200 });

    embedding.backward({ 1, 3, 1 }, dOut);

    // Token 1 appears twice, so its gradient is sum of two rows.
    EXPECT_DOUBLE_EQ(embedding.table.grad[1 * 2 + 0], 101.0);
    EXPECT_DOUBLE_EQ(embedding.table.grad[1 * 2 + 1], 202.0);
    EXPECT_DOUBLE_EQ(embedding.table.grad[3 * 2 + 0], 10.0);
    EXPECT_DOUBLE_EQ(embedding.table.grad[3 * 2 + 1], 20.0);
    // Unused tokens receive nothing.
    EXPECT_DOUBLE_EQ(embedding.table.grad[0], 0.0);
    EXPECT_DOUBLE_EQ(embedding.table.grad[2 * 2 + 0], 0.0);
}

TEST(EmbeddingTest, BackwardKeepsAccumulatingUntilZeroGrad) {
    RandomEngine rng(42);
    Embedding<double> embedding(3, 2, rng);
    auto dOut = makeMatrix(1, 2, { 1, 1 });

    embedding.backward({ 0 }, dOut);
    embedding.backward({ 0 }, dOut);
    EXPECT_DOUBLE_EQ(embedding.table.grad[0], 2.0);

    embedding.zeroGrad();
    EXPECT_DOUBLE_EQ(embedding.table.grad[0], 0.0);
}

TEST(EmbeddingTest, BackwardValidatesShapeAndTokenIds) {
    RandomEngine rng(42);
    Embedding<double> embedding(4, 2, rng);

    // Gradient must be [sequence length, dModel].
    EXPECT_THROW(embedding.backward({ 0, 1 }, makeMatrix(1, 2, { 1, 1 })), InvalidSizeError);
    EXPECT_THROW(embedding.backward({ 0 }, makeMatrix(1, 3, { 1, 1, 1 })), InvalidSizeError);
    EXPECT_THROW(embedding.backward({ 4 }, makeMatrix(1, 2, { 1, 1 })), InvalidParameterError);
}

TEST(EmbeddingTest, UpdateOnlyMovesRowsThatReceivedGradient) {
    RandomEngine rng(42);
    Embedding<double> embedding(3, 2, rng);
    Tensor<double> before = embedding.table.value;
    embedding.backward({ 1 }, makeMatrix(1, 2, { 0.5, -0.5 }));

    embedding.update(0.1);

    EXPECT_DOUBLE_EQ(embedding.table.value[0], before[0]);
    EXPECT_DOUBLE_EQ(embedding.table.value[1], before[1]);
    EXPECT_NEAR(embedding.table.value[2], before[2] - 0.05, 1e-12);
    EXPECT_NEAR(embedding.table.value[3], before[3] + 0.05, 1e-12);
    EXPECT_DOUBLE_EQ(embedding.table.value[4], before[4]);
}

TEST(EmbeddingTest, UpdateRejectsInvalidLearningRate) {
    RandomEngine rng(42);
    Embedding<double> embedding(3, 2, rng);

    EXPECT_THROW(embedding.update(0.0), InvalidParameterError);
}

TEST(EmbeddingTest, SameSeedGivesSameTable) {
    RandomEngine rngA(7), rngB(7);

    Embedding<double> first(6, 4, rngA);
    Embedding<double> second(6, 4, rngB);

    EXPECT_TRUE(testsupport::tensorsEqual(first.table.value, second.table.value));
}

TEST(EmbeddingTest, InitialWeightsAreSmall) {
    RandomEngine rng(42);

    Embedding<double> embedding(50, 16, rng);

    // Scale 0.1: essentially all draws are within a few standard deviations.
    for (std::size_t i = 0; i < embedding.table.value.size(); i++) {
        EXPECT_LT(std::fabs(embedding.table.value[i]), 0.6);
    }
}
