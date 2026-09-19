#include "pch.h"
#include "MultiHeadAttention.h"
#include "TestSupport.h"
#include <cmath>

using testsupport::patternMatrix;
using testsupport::weightedSum;

namespace {
    const std::size_t dModel = 8;
    const std::size_t numHeads = 2;

    // Copy of `x` with its last row changed.
    Tensor<double> withChangedLastRow(const Tensor<double>& x) {
        Tensor<double> changed = x;
        std::size_t last = x.shape[0] - 1;
        for (std::size_t j = 0; j < x.shape[1]; j++) {
            changed[last * x.shape[1] + j] += 1.0 + 0.3 * static_cast<double>(j);
        }
        return changed;
    }
}

TEST(MultiHeadAttentionTest, SplitsModelWidthAcrossHeads) {
    RandomEngine rng(42);

    MultiHeadAttention<double> attention(dModel, numHeads, rng);

    EXPECT_EQ(attention.numHeads, numHeads);
    EXPECT_EQ(attention.dHead, dModel / numHeads);
    EXPECT_EQ(attention.heads.size(), numHeads);
}

TEST(MultiHeadAttentionTest, InvalidConfigurationThrowsInvalidParameterSizeError) {
    RandomEngine rng(42);

    // 8 is not divisible by 3.
    EXPECT_THROW(MultiHeadAttention<double>(8, 3, rng), InvalidParameterSizeError);
    EXPECT_THROW(MultiHeadAttention<double>(8, 0, rng), InvalidParameterSizeError);
    EXPECT_THROW(MultiHeadAttention<double>(0, 2, rng), InvalidParameterSizeError);
}

TEST(MultiHeadAttentionTest, OutputHasSameShapeAsQuery) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    auto x = patternMatrix(5, dModel);
    Tensor<double> out;

    attention.forward(x, x, x, out, true);

    EXPECT_EQ(out.shape, (std::vector<std::size_t>{ 5, dModel }));
}

TEST(MultiHeadAttentionTest, SingleHeadWorks) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, 1, rng);
    auto x = patternMatrix(3, dModel);
    Tensor<double> out;

    EXPECT_NO_THROW(attention.forward(x, x, x, out, true));
    EXPECT_EQ(out.shape, (std::vector<std::size_t>{ 3, dModel }));
}

TEST(MultiHeadAttentionTest, CausalMaskHidesLaterTokens) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    auto x = patternMatrix(5, dModel);
    auto changed = withChangedLastRow(x);
    Tensor<double> before, after;

    attention.forward(x, x, x, before, true);
    attention.forward(changed, changed, changed, after, true);

    for (std::size_t i = 0; i < 4 * dModel; i++) {
        EXPECT_NEAR(before[i], after[i], 1e-12) << "index " << i;
    }
}

TEST(MultiHeadAttentionTest, WithoutMaskEveryTokenSeesEveryOther) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    auto x = patternMatrix(5, dModel);
    auto changed = withChangedLastRow(x);
    Tensor<double> before, after;

    attention.forward(x, x, x, before, false);
    attention.forward(changed, changed, changed, after, false);

    bool firstRowChanged = false;
    for (std::size_t j = 0; j < dModel; j++) {
        firstRowChanged = firstRowChanged || std::fabs(before[j] - after[j]) > 1e-9;
    }
    EXPECT_TRUE(firstRowChanged);
}

TEST(MultiHeadAttentionTest, ForwardRejectsWrongWidth) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    auto wrong = patternMatrix(3, dModel + 2);
    Tensor<double> out;

    EXPECT_THROW(attention.forward(wrong, wrong, wrong, out, true), InvalidSizeError);
}

TEST(MultiHeadAttentionTest, SameSeedGivesIdenticalOutput) {
    RandomEngine rngA(11), rngB(11);
    MultiHeadAttention<double> first(dModel, numHeads, rngA);
    MultiHeadAttention<double> second(dModel, numHeads, rngB);
    auto x = patternMatrix(4, dModel);
    Tensor<double> outA, outB;

    first.forward(x, x, x, outA, true);
    second.forward(x, x, x, outB, true);

    EXPECT_TRUE(testsupport::tensorsEqual(outA, outB));
}

TEST(MultiHeadAttentionTest, BackwardBeforeForwardThrowsInvalidSizeError) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    Tensor<double> dQ, dK, dV;

    EXPECT_THROW(attention.backward(Tensor<double>({ 3, dModel }), dQ, dK, dV), InvalidSizeError);
}

TEST(MultiHeadAttentionTest, BackwardGradientsHaveInputShape) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    auto x = patternMatrix(4, dModel);
    Tensor<double> out, dQ, dK, dV;
    attention.forward(x, x, x, out, true);

    attention.backward(Tensor<double>({ 4, dModel }, 1.0), dQ, dK, dV);

    EXPECT_EQ(dQ.shape, x.shape);
    EXPECT_EQ(dK.shape, x.shape);
    EXPECT_EQ(dV.shape, x.shape);
}

TEST(MultiHeadAttentionTest, SelfAttentionInputGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    Tensor<double> x = patternMatrix(4, dModel);
    Tensor<double> upstream = patternMatrix(4, dModel, 1.0);
    Tensor<double> out, dQ, dK, dV;
    attention.forward(x, x, x, out, true);
    attention.zeroGrad();
    attention.backward(upstream, dQ, dK, dV);
    Tensor<double> dx(x.shape);
    for (std::size_t i = 0; i < dx.size(); i++) {
        dx[i] = dQ[i] + dK[i] + dV[i];
    }

    auto loss = [&]() {
        Tensor<double> y;
        attention.forward(x, x, x, y, true);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss));
}

TEST(MultiHeadAttentionTest, OutputProjectionGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    Tensor<double> x = patternMatrix(4, dModel);
    Tensor<double> upstream = patternMatrix(4, dModel, 1.0);
    Tensor<double> out, dQ, dK, dV;
    attention.forward(x, x, x, out, true);
    attention.zeroGrad();
    attention.backward(upstream, dQ, dK, dV);

    auto loss = [&]() {
        Tensor<double> y;
        attention.forward(x, x, x, y, true);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(attention.Wo.W.value, attention.Wo.W.grad, loss));
}

TEST(MultiHeadAttentionTest, ZeroGradClearsEveryHead) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    auto x = patternMatrix(3, dModel);
    Tensor<double> out, dQ, dK, dV;
    attention.forward(x, x, x, out, true);
    attention.backward(Tensor<double>({ 3, dModel }, 1.0), dQ, dK, dV);

    attention.zeroGrad();

    for (auto& head : attention.heads) {
        for (std::size_t i = 0; i < head.Wq.W.grad.size(); i++) {
            EXPECT_DOUBLE_EQ(head.Wq.W.grad[i], 0.0);
        }
    }
    for (std::size_t i = 0; i < attention.Wo.W.grad.size(); i++) {
        EXPECT_DOUBLE_EQ(attention.Wo.W.grad[i], 0.0);
    }
}

TEST(MultiHeadAttentionTest, UpdateChangesWeights) {
    RandomEngine rng(42);
    MultiHeadAttention<double> attention(dModel, numHeads, rng);
    auto x = patternMatrix(3, dModel);
    Tensor<double> out, dQ, dK, dV;
    attention.forward(x, x, x, out, true);
    attention.zeroGrad();
    attention.backward(Tensor<double>({ 3, dModel }, 0.5), dQ, dK, dV);
    Tensor<double> before = attention.Wo.W.value;

    attention.update(0.01);

    EXPECT_FALSE(testsupport::tensorsEqual(attention.Wo.W.value, before));
}
