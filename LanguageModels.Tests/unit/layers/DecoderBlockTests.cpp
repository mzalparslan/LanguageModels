#include "pch.h"
#include "BasicDecoderBlock.h"
#include "DecoderWithMoe.h"
#include "DecoderMultiHead.h"
#include "DecoderBlockConcept.h"
#include "TestSupport.h"
#include <cmath>

using testsupport::patternMatrix;
using testsupport::tensorsNear;
using testsupport::weightedSum;

namespace {
    const std::size_t dim = 8;
    const std::size_t dFf = 16;

    // Rope tables for a block whose attention head is dim wide.
    class FullWidthRope {
    public:
        static constexpr std::size_t d_head = 8;
        static constexpr std::size_t maxSeqLen = 16;
    };

    Tensor<double> withChangedLastRow(const Tensor<double>& x) {
        Tensor<double> changed = x;
        std::size_t last = x.shape[0] - 1;
        for (std::size_t j = 0; j < x.shape[1]; j++) {
            changed[last * x.shape[1] + j] += 1.0 + 0.3 * static_cast<double>(j);
        }
        return changed;
    }

    // Rows before the last must be identical: a causal block cannot see the future.
    template <typename Block>
    void expectCausal(Block& block) {
        auto x = patternMatrix(5, dim);
        auto changed = withChangedLastRow(x);
        Tensor<double> before, after;

        block.forward(x, before);
        block.forward(changed, after);

        for (std::size_t i = 0; i < 4 * dim; i++) {
            EXPECT_NEAR(before[i], after[i], 1e-12) << "index " << i;
        }
        bool lastRowChanged = false;
        for (std::size_t j = 0; j < dim; j++) {
            lastRowChanged = lastRowChanged || std::fabs(before[4 * dim + j] - after[4 * dim + j]) > 1e-9;
        }
        EXPECT_TRUE(lastRowChanged);
    }

    // Finite-difference check of a block's input gradient.
    template <typename Block>
    void expectInputGradientMatches(Block& block) {
        Tensor<double> x = patternMatrix(4, dim);
        Tensor<double> upstream = patternMatrix(4, dim, 1.0);
        Tensor<double> out, dx;
        block.forward(x, out);
        block.zeroGrad();
        block.backward(upstream, dx);

        auto loss = [&]() {
            Tensor<double> y;
            block.forward(x, y);
            return weightedSum(y, upstream);
        };
        EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss, 1e-4));
    }
}

// === concept

TEST(DecoderBlockConceptTest, AllDecoderBlocksSatisfyTheConcept) {
    static_assert(DecoderBlock<BasicDecoderBlock<double>, double, FullWidthRope>);
    static_assert(DecoderBlock<DecoderWithMoe<double>, double, FullWidthRope>);
    static_assert(DecoderBlock<DecoderMultiHead<double>, double, FullWidthRope>);
    SUCCEED();
}

TEST(DecoderBlockConceptTest, UnrelatedTypesDoNotSatisfyTheConcept) {
    static_assert(!DecoderBlock<int, double, FullWidthRope>);
    static_assert(!DecoderBlock<LinearLayer<double>, double, FullWidthRope>);
    static_assert(!DecoderBlock<FeedForward<double>, double, FullWidthRope>);
    SUCCEED();
}

TEST(DecoderBlockConceptTest, OnlyMultiHeadBlockReportsAnAuxiliaryLoss) {
    static_assert(HasAuxLoss<DecoderMultiHead<double>, double>);
    static_assert(!HasAuxLoss<BasicDecoderBlock<double>, double>);
    static_assert(!HasAuxLoss<DecoderWithMoe<double>, double>);
    SUCCEED();
}

// === BasicDecoderBlock

TEST(BasicDecoderBlockTest, ForwardPreservesShape) {
    RandomEngine rng(42);
    BasicDecoderBlock<double> block(dim, dFf, rng);
    auto x = patternMatrix(5, dim);
    Tensor<double> out;

    block.forward(x, out);

    EXPECT_EQ(out.shape, x.shape);
    EXPECT_FALSE(tensorsNear(out, x, 1e-12));
    EXPECT_NO_THROW(validation::requireAllFinite(out, "block output"));
}

TEST(BasicDecoderBlockTest, ForwardRejectsWrongWidth) {
    RandomEngine rng(42);
    BasicDecoderBlock<double> block(dim, dFf, rng);
    Tensor<double> out;

    EXPECT_THROW(block.forward(patternMatrix(3, dim + 1), out), InvalidSizeError);
}

TEST(BasicDecoderBlockTest, IsCausal) {
    RandomEngine rng(42);
    BasicDecoderBlock<double> block(dim, dFf, rng);

    expectCausal(block);
}

TEST(BasicDecoderBlockTest, IsCausalWithRotaryEmbedding) {
    RandomEngine rng(42);
    BasicDecoderBlock<double> block(dim, dFf, rng);
    RotaryEmbedding<double, FullWidthRope> rope;
    auto x = patternMatrix(5, dim);
    auto changed = withChangedLastRow(x);
    Tensor<double> before, after;

    block.forward(x, before, &rope);
    block.forward(changed, after, &rope);

    for (std::size_t i = 0; i < 4 * dim; i++) {
        EXPECT_NEAR(before[i], after[i], 1e-12) << "index " << i;
    }
}

TEST(BasicDecoderBlockTest, SameSeedGivesIdenticalOutput) {
    RandomEngine rngA(8), rngB(8);
    BasicDecoderBlock<double> first(dim, dFf, rngA);
    BasicDecoderBlock<double> second(dim, dFf, rngB);
    auto x = patternMatrix(4, dim);
    Tensor<double> outA, outB;

    first.forward(x, outA);
    second.forward(x, outB);

    EXPECT_TRUE(testsupport::tensorsEqual(outA, outB));
}

TEST(BasicDecoderBlockTest, InputGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    BasicDecoderBlock<double> block(dim, dFf, rng);

    expectInputGradientMatches(block);
}

TEST(BasicDecoderBlockTest, InputGradientMatchesFiniteDifferencesWithRotaryEmbedding) {
    RandomEngine rng(42);
    BasicDecoderBlock<double> block(dim, dFf, rng);
    RotaryEmbedding<double, FullWidthRope> rope;
    Tensor<double> x = patternMatrix(4, dim);
    Tensor<double> upstream = patternMatrix(4, dim, 1.0);
    Tensor<double> out, dx;
    block.forward(x, out, &rope);
    block.zeroGrad();
    block.backward(upstream, dx);

    auto loss = [&]() {
        Tensor<double> y;
        block.forward(x, y, &rope);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss, 1e-4));
}

TEST(BasicDecoderBlockTest, ZeroGradClearsNormGradients) {
    RandomEngine rng(42);
    BasicDecoderBlock<double> block(dim, dFf, rng);
    auto x = patternMatrix(3, dim);
    Tensor<double> out, dx;
    block.forward(x, out);
    block.backward(patternMatrix(3, dim, 1.0), dx);
    ASSERT_NE(block.norm1.weight.grad[0], 0.0);

    block.zeroGrad();

    for (std::size_t i = 0; i < dim; i++) {
        EXPECT_DOUBLE_EQ(block.norm1.weight.grad[i], 0.0);
        EXPECT_DOUBLE_EQ(block.norm2.weight.grad[i], 0.0);
    }
}

TEST(BasicDecoderBlockTest, UpdateChangesWeights) {
    RandomEngine rng(42);
    BasicDecoderBlock<double> block(dim, dFf, rng);
    auto x = patternMatrix(3, dim);
    Tensor<double> out, dx;
    block.forward(x, out);
    block.zeroGrad();
    block.backward(patternMatrix(3, dim, 1.0), dx);
    Tensor<double> before = block.norm1.weight.value;

    block.update(0.01);

    EXPECT_FALSE(testsupport::tensorsEqual(block.norm1.weight.value, before));
}

// === DecoderWithMoe

TEST(DecoderWithMoeTest, DenseModeOwnsFeedForwardOnly) {
    RandomEngine rng(42);

    DecoderWithMoe<double> block(dim, dFf, rng, false);

    EXPECT_NE(block.ff, nullptr);
    EXPECT_EQ(block.moe, nullptr);
}

TEST(DecoderWithMoeTest, MixtureModeOwnsMoeOnly) {
    RandomEngine rng(42);

    DecoderWithMoe<double> block(dim, dFf, rng, true, 4, 2);

    EXPECT_EQ(block.ff, nullptr);
    ASSERT_NE(block.moe, nullptr);
    EXPECT_EQ(block.moe->numExperts, 4u);
    EXPECT_EQ(block.moe->topK, 2u);
}

TEST(DecoderWithMoeTest, TooManyExpertsPerTokenThrowsInvalidParameterSizeError) {
    RandomEngine rng(42);

    EXPECT_THROW(DecoderWithMoe<double>(dim, dFf, rng, true, 2, 3), InvalidParameterSizeError);
}

TEST(DecoderWithMoeTest, ForwardPreservesShapeInBothModes) {
    RandomEngine rng(42);
    DecoderWithMoe<double> dense(dim, dFf, rng, false);
    DecoderWithMoe<double> mixture(dim, dFf, rng, true, 4, 2);
    auto x = patternMatrix(5, dim);
    Tensor<double> denseOut, mixtureOut;

    dense.forward(x, denseOut);
    mixture.forward(x, mixtureOut);

    EXPECT_EQ(denseOut.shape, x.shape);
    EXPECT_EQ(mixtureOut.shape, x.shape);
    EXPECT_NO_THROW(validation::requireAllFinite(mixtureOut, "block output"));
}

TEST(DecoderWithMoeTest, ForwardRejectsWrongWidth) {
    RandomEngine rng(42);
    DecoderWithMoe<double> block(dim, dFf, rng, true);
    Tensor<double> out;

    EXPECT_THROW(block.forward(patternMatrix(3, dim - 1), out), InvalidSizeError);
}

TEST(DecoderWithMoeTest, DenseModeIsCausal) {
    RandomEngine rng(42);
    DecoderWithMoe<double> block(dim, dFf, rng, false);

    expectCausal(block);
}

TEST(DecoderWithMoeTest, MixtureModeIsCausal) {
    RandomEngine rng(42);
    DecoderWithMoe<double> block(dim, dFf, rng, true, 4, 2);

    expectCausal(block);
}

TEST(DecoderWithMoeTest, DenseInputGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    DecoderWithMoe<double> block(dim, dFf, rng, false);

    expectInputGradientMatches(block);
}

TEST(DecoderWithMoeTest, MixtureBackwardProducesInputShapedFiniteGradient) {
    RandomEngine rng(42);
    DecoderWithMoe<double> block(dim, dFf, rng, true, 4, 2);
    auto x = patternMatrix(4, dim);
    Tensor<double> out, dx;
    block.forward(x, out);
    block.zeroGrad();

    block.backward(patternMatrix(4, dim, 1.0), dx);

    EXPECT_EQ(dx.shape, x.shape);
    EXPECT_NO_THROW(validation::requireAllFinite(dx, "input gradient"));
}

TEST(DecoderWithMoeTest, UpdateWorksInBothModes) {
    RandomEngine rng(42);
    DecoderWithMoe<double> dense(dim, dFf, rng, false);
    DecoderWithMoe<double> mixture(dim, dFf, rng, true, 4, 2);
    auto x = patternMatrix(3, dim);
    Tensor<double> out, dx;

    for (auto* block : { &dense, &mixture }) {
        block->forward(x, out);
        block->zeroGrad();
        block->backward(patternMatrix(3, dim, 1.0), dx);
        EXPECT_NO_THROW(block->update(0.01));
        EXPECT_NO_THROW(block->update(0.01, UpdateRule::adam(1)));
    }
}

// === DecoderMultiHead

TEST(DecoderMultiHeadTest, ForwardPreservesShapeInBothModes) {
    RandomEngine rng(42);
    DecoderMultiHead<double> dense(dim, dFf, rng, false);
    DecoderMultiHead<double> mixture(dim, dFf, rng, true, 4, 2);
    auto x = patternMatrix(5, dim);
    Tensor<double> denseOut, mixtureOut;

    dense.forward(x, denseOut);
    mixture.forward(x, mixtureOut);

    EXPECT_EQ(denseOut.shape, x.shape);
    EXPECT_EQ(mixtureOut.shape, x.shape);
}

TEST(DecoderMultiHeadTest, HeadCountMustDivideModelWidth) {
    RandomEngine rng(42);

    EXPECT_THROW(DecoderMultiHead<double>(dim, dFf, rng, false, 4, 2, 3), InvalidParameterSizeError);
    EXPECT_NO_THROW(DecoderMultiHead<double>(dim, dFf, rng, false, 4, 2, 2));
}

TEST(DecoderMultiHeadTest, ForwardRejectsWrongWidth) {
    RandomEngine rng(42);
    DecoderMultiHead<double> block(dim, dFf, rng);
    Tensor<double> out;

    EXPECT_THROW(block.forward(patternMatrix(3, dim + 1), out), InvalidSizeError);
}

TEST(DecoderMultiHeadTest, IsCausal) {
    RandomEngine rng(42);
    DecoderMultiHead<double> block(dim, dFf, rng, false);

    expectCausal(block);
}

TEST(DecoderMultiHeadTest, DenseInputGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    DecoderMultiHead<double> block(dim, dFf, rng, false);

    expectInputGradientMatches(block);
}

TEST(DecoderMultiHeadTest, AuxiliaryLossIsZeroWithoutMoe) {
    RandomEngine rng(42);
    DecoderMultiHead<double> block(dim, dFf, rng, false);

    EXPECT_DOUBLE_EQ(block.getAuxLoss(), 0.0);
}
