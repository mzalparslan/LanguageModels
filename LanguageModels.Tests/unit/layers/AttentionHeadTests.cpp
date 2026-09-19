#include "pch.h"
#include "AttentionHead.h"
#include "TestSupport.h"
#include <cmath>

using testsupport::patternMatrix;
using testsupport::tensorsNear;
using testsupport::weightedSum;

namespace {
    // RoPE tables sized for a 4-wide head (matches AttentionHead(8, 4)).
    class HeadRopeConfig {
    public:
        static constexpr std::size_t d_head = 4;
        static constexpr std::size_t maxSeqLen = 16;
    };

    const std::size_t dModel = 8;
    const std::size_t dHead = 4;

    // Copy of `x` with row `row` replaced by different values.
    Tensor<double> withChangedRow(const Tensor<double>& x, std::size_t row) {
        Tensor<double> changed = x;
        for (std::size_t j = 0; j < x.shape[1]; j++) {
            changed[row * x.shape[1] + j] += 1.0 + 0.3 * static_cast<double>(j);
        }
        return changed;
    }

    // Typed null pointer, so `rope` can be omitted while still passing `masked`.
    RotaryEmbedding<double, HeadRopeConfig>* const noRope = nullptr;
}

TEST(AttentionHeadTest, SelfAttentionOutputHasOneRowPerQueryAndHeadWidth) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto x = patternMatrix(5, dModel);
    Tensor<double> out;

    head.forward(x, x, x, out, noRope, false);

    EXPECT_EQ(out.shape, (std::vector<std::size_t>{ 5, dHead }));
}

TEST(AttentionHeadTest, ShortConstructorUsesFullModelWidth) {
    RandomEngine rng(42);

    AttentionHead<double> head(dModel, rng);

    EXPECT_EQ(head.dHead, dModel);
}

TEST(AttentionHeadTest, AttentionWeightsFormAProbabilityDistributionPerRow) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto x = patternMatrix(5, dModel);
    Tensor<double> out;

    head.forward(x, x, x, out, noRope, false);

    ASSERT_EQ(head.attnWeights.shape, (std::vector<std::size_t>{ 5, 5 }));
    for (std::size_t i = 0; i < 5; i++) {
        double sum = 0.0;
        for (std::size_t j = 0; j < 5; j++) {
            EXPECT_GT(head.attnWeights[i * 5 + j], 0.0);
            sum += head.attnWeights[i * 5 + j];
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST(AttentionHeadTest, CausalMaskZeroesFutureWeights) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto x = patternMatrix(5, dModel);
    Tensor<double> out;

    head.forward(x, x, x, out, noRope, true);

    for (std::size_t i = 0; i < 5; i++) {
        double sum = 0.0;
        for (std::size_t j = 0; j < 5; j++) {
            if (j > i) {
                EXPECT_DOUBLE_EQ(head.attnWeights[i * 5 + j], 0.0);
            }
            sum += head.attnWeights[i * 5 + j];
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
    // The first token can only attend to itself.
    EXPECT_NEAR(head.attnWeights[0], 1.0, 1e-12);
}

TEST(AttentionHeadTest, CausalOutputIgnoresLaterTokens) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto x = patternMatrix(5, dModel);
    auto changed = withChangedRow(x, 4);
    Tensor<double> before, after;

    head.forward(x, x, x, before, noRope, true);
    head.forward(changed, changed, changed, after, noRope, true);

    // Rows 0..3 must not see the modified last token; row 4 must.
    for (std::size_t i = 0; i < 4 * dHead; i++) {
        EXPECT_NEAR(before[i], after[i], 1e-12) << "index " << i;
    }
    bool lastRowChanged = false;
    for (std::size_t j = 0; j < dHead; j++) {
        lastRowChanged = lastRowChanged || std::fabs(before[4 * dHead + j] - after[4 * dHead + j]) > 1e-9;
    }
    EXPECT_TRUE(lastRowChanged);
}

TEST(AttentionHeadTest, UnmaskedOutputSeesEveryToken) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto x = patternMatrix(5, dModel);
    auto changed = withChangedRow(x, 4);
    Tensor<double> before, after;

    head.forward(x, x, x, before, noRope, false);
    head.forward(changed, changed, changed, after, noRope, false);

    bool firstRowChanged = false;
    for (std::size_t j = 0; j < dHead; j++) {
        firstRowChanged = firstRowChanged || std::fabs(before[j] - after[j]) > 1e-9;
    }
    EXPECT_TRUE(firstRowChanged);
}

TEST(AttentionHeadTest, CrossAttentionOutputFollowsQueryLength) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto query = patternMatrix(2, dModel);
    auto memory = patternMatrix(6, dModel, 1.0);
    Tensor<double> out;

    head.forward(query, memory, memory, out, noRope, false);

    EXPECT_EQ(out.shape, (std::vector<std::size_t>{ 2, dHead }));
    EXPECT_EQ(head.attnWeights.shape, (std::vector<std::size_t>{ 2, 6 }));
}

TEST(AttentionHeadTest, ForwardValidatesInputs) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto good = patternMatrix(4, dModel);
    auto wrongWidth = patternMatrix(4, dModel + 1);
    auto differentLength = patternMatrix(3, dModel);
    Tensor<double> out;

    EXPECT_THROW(head.forward(wrongWidth, good, good, out, noRope, false), InvalidSizeError);
    EXPECT_THROW(head.forward(good, wrongWidth, good, out, noRope, false), InvalidSizeError);
    EXPECT_THROW(head.forward(good, good, wrongWidth, out, noRope, false), InvalidSizeError);
    // Keys and values must have the same number of rows.
    EXPECT_THROW(head.forward(good, good, differentLength, out, noRope, false), InvalidSizeError);
    EXPECT_THROW(head.forward(Tensor<double>({ dModel }), good, good, out, noRope, false), InvalidSizeError);
}

TEST(AttentionHeadTest, SameSeedGivesIdenticalOutput) {
    RandomEngine rngA(5), rngB(5);
    AttentionHead<double> first(dModel, dHead, rngA);
    AttentionHead<double> second(dModel, dHead, rngB);
    auto x = patternMatrix(4, dModel);
    Tensor<double> outA, outB;

    first.forward(x, x, x, outA, noRope, true);
    second.forward(x, x, x, outB, noRope, true);

    EXPECT_TRUE(testsupport::tensorsEqual(outA, outB));
}

TEST(AttentionHeadRopeTest, SingleTokenIsUnaffectedByRotation) {
    // Position 0 is not rotated, so a 1-token sequence must match the RoPE-free result.
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    RotaryEmbedding<double, HeadRopeConfig> rope;
    auto x = patternMatrix(1, dModel);
    Tensor<double> plain, rotated;

    head.forward(x, x, x, plain, noRope, true);
    head.forward(x, x, x, rotated, &rope, true);

    EXPECT_TRUE(tensorsNear(rotated, plain, 1e-12));
}

TEST(AttentionHeadRopeTest, RotationChangesMultiTokenOutput) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    RotaryEmbedding<double, HeadRopeConfig> rope;
    auto x = patternMatrix(5, dModel);
    Tensor<double> plain, rotated;

    head.forward(x, x, x, plain, noRope, true);
    head.forward(x, x, x, rotated, &rope, true);

    EXPECT_FALSE(tensorsNear(rotated, plain, 1e-9));
}

TEST(AttentionHeadRopeTest, RotationStillKeepsCausality) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    RotaryEmbedding<double, HeadRopeConfig> rope;
    auto x = patternMatrix(5, dModel);
    auto changed = withChangedRow(x, 4);
    Tensor<double> before, after;

    head.forward(x, x, x, before, &rope, true);
    head.forward(changed, changed, changed, after, &rope, true);

    for (std::size_t i = 0; i < 4 * dHead; i++) {
        EXPECT_NEAR(before[i], after[i], 1e-12) << "index " << i;
    }
}

// ------------------------------------------------------------------- backward

TEST(AttentionHeadBackwardTest, BackwardBeforeForwardThrowsInvalidSizeError) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    Tensor<double> dQ, dK, dV;

    EXPECT_THROW(head.backward(Tensor<double>({ 3, dHead }), dQ, dK, dV), InvalidSizeError);
}

TEST(AttentionHeadBackwardTest, BackwardRejectsGradientOfWrongShape) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto x = patternMatrix(3, dModel);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, noRope, true);

    EXPECT_THROW(head.backward(Tensor<double>({ 2, dHead }, 1.0), dQ, dK, dV), InvalidSizeError);
    EXPECT_THROW(head.backward(Tensor<double>({ 3, dHead + 1 }, 1.0), dQ, dK, dV), InvalidSizeError);
}

TEST(AttentionHeadBackwardTest, InputGradientsHaveInputShape) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto x = patternMatrix(3, dModel);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, noRope, true);

    head.backward(Tensor<double>({ 3, dHead }, 1.0), dQ, dK, dV);

    EXPECT_EQ(dQ.shape, x.shape);
    EXPECT_EQ(dK.shape, x.shape);
    EXPECT_EQ(dV.shape, x.shape);
}

TEST(AttentionHeadBackwardTest, SelfAttentionInputGradientMatchesFiniteDifferences) {
    // For self-attention the same tensor feeds Q, K and V, so the total input
    // gradient is dQ + dK + dV.
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    Tensor<double> x = patternMatrix(4, dModel);
    Tensor<double> upstream = patternMatrix(4, dHead, 1.0);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, noRope, true);
    head.zeroGrad();
    head.backward(upstream, dQ, dK, dV);
    Tensor<double> dx(x.shape);
    for (std::size_t i = 0; i < dx.size(); i++) {
        dx[i] = dQ[i] + dK[i] + dV[i];
    }

    auto loss = [&]() {
        Tensor<double> y;
        head.forward(x, x, x, y, noRope, true);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss));
}

TEST(AttentionHeadBackwardTest, UnmaskedSelfAttentionInputGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    Tensor<double> x = patternMatrix(4, dModel);
    Tensor<double> upstream = patternMatrix(4, dHead, 1.0);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, noRope, false);
    head.zeroGrad();
    head.backward(upstream, dQ, dK, dV);
    Tensor<double> dx(x.shape);
    for (std::size_t i = 0; i < dx.size(); i++) {
        dx[i] = dQ[i] + dK[i] + dV[i];
    }

    auto loss = [&]() {
        Tensor<double> y;
        head.forward(x, x, x, y, noRope, false);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss));
}

TEST(AttentionHeadBackwardTest, ProjectionWeightGradientsMatchFiniteDifferences) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    Tensor<double> x = patternMatrix(4, dModel);
    Tensor<double> upstream = patternMatrix(4, dHead, 1.0);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, noRope, true);
    head.zeroGrad();
    head.backward(upstream, dQ, dK, dV);

    auto loss = [&]() {
        Tensor<double> y;
        head.forward(x, x, x, y, noRope, true);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(head.Wq.W.value, head.Wq.W.grad, loss)) << "Wq";
    EXPECT_TRUE(testsupport::gradientMatches(head.Wk.W.value, head.Wk.W.grad, loss)) << "Wk";
    EXPECT_TRUE(testsupport::gradientMatches(head.Wv.W.value, head.Wv.W.grad, loss)) << "Wv";
    EXPECT_TRUE(testsupport::gradientMatches(head.Wo.W.value, head.Wo.W.grad, loss)) << "Wo";
}

TEST(AttentionHeadBackwardTest, UpdateChangesAllFourProjections) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    auto x = patternMatrix(4, dModel);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, noRope, true);
    head.zeroGrad();
    head.backward(Tensor<double>({ 4, dHead }, 0.5), dQ, dK, dV);
    Tensor<double> wq = head.Wq.W.value, wk = head.Wk.W.value;
    Tensor<double> wv = head.Wv.W.value, wo = head.Wo.W.value;

    head.update(0.01);

    EXPECT_FALSE(testsupport::tensorsEqual(head.Wq.W.value, wq));
    EXPECT_FALSE(testsupport::tensorsEqual(head.Wk.W.value, wk));
    EXPECT_FALSE(testsupport::tensorsEqual(head.Wv.W.value, wv));
    EXPECT_FALSE(testsupport::tensorsEqual(head.Wo.W.value, wo));
}

// With RoPE, Q and K are rotated after projection, so backward() must rotate
// their gradients back before they reach Wq/Wk.
TEST(AttentionHeadRopeBackwardTest, QueryKeyWeightGradientsMatchFiniteDifferences) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    RotaryEmbedding<double, HeadRopeConfig> rope;
    Tensor<double> x = patternMatrix(4, dModel);
    Tensor<double> upstream = patternMatrix(4, dHead, 1.0);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, &rope, true);
    head.zeroGrad();
    head.backward(upstream, dQ, dK, dV);

    auto loss = [&]() {
        Tensor<double> y;
        head.forward(x, x, x, y, &rope, true);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(head.Wq.W.value, head.Wq.W.grad, loss)) << "Wq";
    EXPECT_TRUE(testsupport::gradientMatches(head.Wk.W.value, head.Wk.W.grad, loss)) << "Wk";
    EXPECT_TRUE(testsupport::gradientMatches(head.Wv.W.value, head.Wv.W.grad, loss)) << "Wv";
    EXPECT_TRUE(testsupport::gradientMatches(head.Wo.W.value, head.Wo.W.grad, loss)) << "Wo";
}

TEST(AttentionHeadRopeBackwardTest, InputGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    RotaryEmbedding<double, HeadRopeConfig> rope;
    Tensor<double> x = patternMatrix(4, dModel);
    Tensor<double> upstream = patternMatrix(4, dHead, 1.0);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, &rope, true);
    head.zeroGrad();
    head.backward(upstream, dQ, dK, dV);
    Tensor<double> dx(x.shape);
    for (std::size_t i = 0; i < dx.size(); i++) {
        dx[i] = dQ[i] + dK[i] + dV[i];
    }

    auto loss = [&]() {
        Tensor<double> y;
        head.forward(x, x, x, y, &rope, true);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss));
}

TEST(AttentionHeadRopeBackwardTest, CrossAttentionGradientsMatchFiniteDifferences) {
    // Different query and key lengths exercise the two rotations separately.
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    RotaryEmbedding<double, HeadRopeConfig> rope;
    Tensor<double> query = patternMatrix(3, dModel);
    Tensor<double> memory = patternMatrix(5, dModel, 1.0);
    Tensor<double> upstream = patternMatrix(3, dHead, 2.0);
    Tensor<double> out, dQ, dK, dV;
    head.forward(query, memory, memory, out, &rope, false);
    head.zeroGrad();
    head.backward(upstream, dQ, dK, dV);

    auto loss = [&]() {
        Tensor<double> y;
        head.forward(query, memory, memory, y, &rope, false);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(query, dQ, loss)) << "query";
    EXPECT_TRUE(testsupport::gradientMatches(head.Wk.W.value, head.Wk.W.grad, loss)) << "Wk";
}

TEST(AttentionHeadRopeBackwardTest, ForwardWithoutRopeAfterForwardWithRopeUsesPlainGradients) {
    // The "was RoPE used" state belongs to the most recent forward().
    RandomEngine rng(42);
    AttentionHead<double> head(dModel, dHead, rng);
    RotaryEmbedding<double, HeadRopeConfig> rope;
    Tensor<double> x = patternMatrix(4, dModel);
    Tensor<double> upstream = patternMatrix(4, dHead, 1.0);
    Tensor<double> out, dQ, dK, dV;
    head.forward(x, x, x, out, &rope, true);
    head.forward(x, x, x, out, noRope, true);
    head.zeroGrad();
    head.backward(upstream, dQ, dK, dV);

    auto loss = [&]() {
        Tensor<double> y;
        head.forward(x, x, x, y, noRope, true);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(head.Wq.W.value, head.Wq.W.grad, loss));
}
