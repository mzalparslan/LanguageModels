#include "pch.h"
#include "MoELayer.h"
#include "TestSupport.h"
#include <algorithm>
#include <set>

using testsupport::patternMatrix;

namespace {
    const std::size_t dModel = 6;
    const std::size_t dFf = 12;
}

TEST(MoELayerTest, BuildsOneFeedForwardPerExpert) {
    RandomEngine rng(42);

    MoELayer<double> moe(dModel, dFf, rng, 5, 2);

    EXPECT_EQ(moe.experts.size(), 5u);
    EXPECT_EQ(moe.numExperts, 5u);
    EXPECT_EQ(moe.topK, 2u);
}

TEST(MoELayerTest, DefaultsToFourExpertsAndTwoPerToken) {
    RandomEngine rng(42);

    MoELayer<double> moe(dModel, dFf, rng);

    EXPECT_EQ(moe.numExperts, 4u);
    EXPECT_EQ(moe.topK, 2u);
}

TEST(MoELayerTest, InvalidConfigurationThrowsInvalidParameterSizeError) {
    RandomEngine rng(42);

    EXPECT_THROW(MoELayer<double>(dModel, dFf, rng, 0, 1), InvalidParameterSizeError);
    EXPECT_THROW(MoELayer<double>(dModel, dFf, rng, 4, 0), InvalidParameterSizeError);
    EXPECT_THROW(MoELayer<double>(dModel, dFf, rng, 2, 3), InvalidParameterSizeError);
    EXPECT_THROW(MoELayer<double>(0, dFf, rng, 4, 2), InvalidParameterSizeError);
}

TEST(MoELayerTest, ForwardPreservesShape) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng, 4, 2);
    auto x = patternMatrix(5, dModel);
    Tensor<double> out;

    moe.forward(x, out);

    EXPECT_EQ(out.shape, x.shape);
}

TEST(MoELayerTest, ForwardRejectsWrongWidth) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng);
    Tensor<double> out;

    EXPECT_THROW(moe.forward(patternMatrix(3, dModel + 1), out), InvalidSizeError);
    EXPECT_THROW(moe.forward(Tensor<double>({ dModel }), out), InvalidSizeError);
}

TEST(MoELayerTest, RoutesEveryTokenToDistinctTopKExperts) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng, 6, 3);
    auto x = patternMatrix(7, dModel);
    Tensor<double> out;

    moe.forward(x, out);

    ASSERT_EQ(moe.selectedIndices.size(), 7u);
    for (const auto& row : moe.selectedIndices) {
        ASSERT_EQ(row.size(), 3u);
        std::set<int> distinct(row.begin(), row.end());
        EXPECT_EQ(distinct.size(), 3u);
        for (int expert : row) {
            EXPECT_GE(expert, 0);
            EXPECT_LT(expert, 6);
        }
    }
}

TEST(MoELayerTest, SelectedWeightsAreRenormalizedAndSortedDescending) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng, 6, 3);
    auto x = patternMatrix(7, dModel);
    Tensor<double> out;

    moe.forward(x, out);

    for (const auto& weights : moe.selectedWeights) {
        double sum = 0.0;
        for (std::size_t k = 0; k < weights.size(); k++) {
            EXPECT_GT(weights[k], 0.0);
            sum += weights[k];
            if (k > 0) {
                EXPECT_GE(weights[k - 1], weights[k]);
            }
        }
        EXPECT_NEAR(sum, 1.0, 1e-12);
    }
}

TEST(MoELayerTest, OutputIsTheWeightedSumOfTheSelectedExperts) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng, 4, 2);
    auto x = patternMatrix(3, dModel);
    Tensor<double> out;
    moe.forward(x, out);

    for (std::size_t token = 0; token < 3; token++) {
        Tensor<double> row({ 1, dModel });
        for (std::size_t j = 0; j < dModel; j++) {
            row[j] = x[token * dModel + j];
        }
        std::vector<double> expected(dModel, 0.0);
        for (std::size_t k = 0; k < moe.topK; k++) {
            Tensor<double> expertOut;
            moe.experts[moe.selectedIndices[token][k]].forward(row, expertOut);
            for (std::size_t j = 0; j < dModel; j++) {
                expected[j] += moe.selectedWeights[token][k] * expertOut[j];
            }
        }
        for (std::size_t j = 0; j < dModel; j++) {
            EXPECT_NEAR(out[token * dModel + j], expected[j], 1e-12);
        }
    }
}

TEST(MoELayerTest, RoutingUsesEveryExpertWhenTopKEqualsExpertCount) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng, 3, 3);
    auto x = patternMatrix(4, dModel);
    Tensor<double> out;

    moe.forward(x, out);

    for (const auto& row : moe.selectedIndices) {
        std::set<int> distinct(row.begin(), row.end());
        EXPECT_EQ(distinct, (std::set<int>{ 0, 1, 2 }));
    }
}

TEST(MoELayerTest, SingleExpertPassesFullWeightToIt) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng, 1, 1);
    auto x = patternMatrix(2, dModel);
    Tensor<double> out;

    moe.forward(x, out);

    for (const auto& weights : moe.selectedWeights) {
        EXPECT_NEAR(weights[0], 1.0, 1e-12);
    }
}

TEST(MoELayerTest, SameSeedGivesIdenticalOutput) {
    RandomEngine rngA(4), rngB(4);
    MoELayer<double> first(dModel, dFf, rngA);
    MoELayer<double> second(dModel, dFf, rngB);
    auto x = patternMatrix(3, dModel);
    Tensor<double> outA, outB;

    first.forward(x, outA);
    second.forward(x, outB);

    EXPECT_TRUE(testsupport::tensorsEqual(outA, outB));
}

TEST(MoELayerTest, BackwardBeforeForwardThrowsInvalidSizeError) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng);
    Tensor<double> dx;

    EXPECT_THROW(moe.backward(Tensor<double>({ 2, dModel }), dx), InvalidSizeError);
}

TEST(MoELayerTest, BackwardRejectsGradientOfWrongShape) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng);
    Tensor<double> out, dx;
    moe.forward(patternMatrix(3, dModel), out);

    EXPECT_THROW(moe.backward(Tensor<double>({ 2, dModel }, 1.0), dx), InvalidSizeError);
    EXPECT_THROW(moe.backward(Tensor<double>({ 3, dModel + 1 }, 1.0), dx), InvalidSizeError);
}

TEST(MoELayerTest, BackwardProducesFiniteNonZeroInputGradient) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng, 4, 2);
    auto x = patternMatrix(4, dModel);
    Tensor<double> out, dx;
    moe.forward(x, out);
    moe.zeroGrad();

    moe.backward(patternMatrix(4, dModel, 1.0), dx);

    EXPECT_EQ(dx.shape, x.shape);
    EXPECT_NO_THROW(validation::requireAllFinite(dx, "input gradient"));
    bool anyNonZero = false;
    for (std::size_t i = 0; i < dx.size(); i++) {
        anyNonZero = anyNonZero || dx[i] != 0.0;
    }
    EXPECT_TRUE(anyNonZero);
}

TEST(MoELayerTest, ZeroGradAndUpdateRunOnEveryExpert) {
    RandomEngine rng(42);
    MoELayer<double> moe(dModel, dFf, rng, 4, 2);
    auto x = patternMatrix(4, dModel);
    Tensor<double> out, dx;
    moe.forward(x, out);
    moe.zeroGrad();
    moe.backward(patternMatrix(4, dModel, 1.0), dx);

    EXPECT_NO_THROW(moe.update(0.01));
    EXPECT_NO_THROW(moe.update(0.01, UpdateRule::adam(1)));
    EXPECT_NO_THROW(moe.zeroGrad());
}
