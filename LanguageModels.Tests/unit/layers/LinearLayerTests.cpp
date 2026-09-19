#include "pch.h"
#include "LinearLayer.h"
#include "TestSupport.h"
#include <cmath>

using testsupport::makeMatrix;
using testsupport::patternMatrix;
using testsupport::tensorsNear;
using testsupport::weightedSum;

namespace {
    // A 3 -> 2 layer with hand-picked weights so results can be checked by hand.
    LinearLayer<double> makeKnownLayer(RandomEngine& rng) {
        LinearLayer<double> layer(3, 2, rng);
        layer.W.value = makeMatrix(3, 2, { 1, 2, 3, 4, 5, 6 });
        layer.b.value[0] = 0.5;
        layer.b.value[1] = -0.5;
        return layer;
    }
}

TEST(LinearLayerTest, WeightsAndBiasHaveExpectedShapes) {
    RandomEngine rng(42);

    LinearLayer<double> layer(3, 5, rng);

    EXPECT_EQ(layer.W.value.shape, (std::vector<std::size_t>{ 3, 5 }));
    EXPECT_EQ(layer.b.value.shape, (std::vector<std::size_t>{ 5 }));
    EXPECT_EQ(layer.dIn, 3u);
    EXPECT_EQ(layer.dOut, 5u);
}

TEST(LinearLayerTest, BiasStartsAtZero) {
    RandomEngine rng(42);

    LinearLayer<double> layer(3, 5, rng);

    for (std::size_t i = 0; i < 5; i++) {
        EXPECT_DOUBLE_EQ(layer.b.value[i], 0.0);
    }
}

TEST(LinearLayerTest, WeightScaleFollowsHeInitialization) {
    RandomEngine rng(42);
    const std::size_t dIn = 100;

    LinearLayer<double> layer(dIn, 100, rng);

    // He init: weights ~ N(0, 2 / dIn), so the sample variance is near 0.02.
    double sumSquares = 0.0;
    for (std::size_t i = 0; i < layer.W.value.size(); i++) {
        sumSquares += layer.W.value[i] * layer.W.value[i];
    }
    double variance = sumSquares / static_cast<double>(layer.W.value.size());
    EXPECT_NEAR(variance, 2.0 / dIn, 0.1 * 2.0 / dIn);
}

TEST(LinearLayerTest, ZeroSizesThrowInvalidParameterSizeError) {
    RandomEngine rng(42);

    EXPECT_THROW(LinearLayer<double>(0, 4, rng), InvalidParameterSizeError);
    EXPECT_THROW(LinearLayer<double>(4, 0, rng), InvalidParameterSizeError);
}

TEST(LinearLayerTest, SameSeedGivesIdenticalLayers) {
    RandomEngine rngA(9), rngB(9);

    LinearLayer<double> first(4, 4, rngA);
    LinearLayer<double> second(4, 4, rngB);

    EXPECT_TRUE(testsupport::tensorsEqual(first.W.value, second.W.value));
}

TEST(LinearLayerTest, ForwardComputesAffineTransform) {
    RandomEngine rng(42);
    auto layer = makeKnownLayer(rng);
    auto x = makeMatrix(2, 3, { 1, 0, 2, 0, 1, -1 });
    Tensor<double> out;

    layer.forward(x, out);

    // Row 0: [1*1 + 0*3 + 2*5 + 0.5, 1*2 + 0*4 + 2*6 - 0.5] = [11.5, 13.5]
    // Row 1: [0*1 + 1*3 - 1*5 + 0.5, 0*2 + 1*4 - 1*6 - 0.5] = [-1.5, -2.5]
    EXPECT_TRUE(tensorsNear(out, makeMatrix(2, 2, { 11.5, 13.5, -1.5, -2.5 }), 1e-12));
}

TEST(LinearLayerTest, ForwardIsIndependentPerRow) {
    RandomEngine rng(42);
    auto layer = makeKnownLayer(rng);
    auto both = makeMatrix(2, 3, { 1, 0, 2, 0, 1, -1 });
    auto second = makeMatrix(1, 3, { 0, 1, -1 });
    Tensor<double> outBoth, outSecond;

    layer.forward(both, outBoth);
    layer.forward(second, outSecond);

    EXPECT_NEAR(outBoth[2], outSecond[0], 1e-12);
    EXPECT_NEAR(outBoth[3], outSecond[1], 1e-12);
}

TEST(LinearLayerTest, ForwardRejectsWrongWidthAndNonMatrix) {
    RandomEngine rng(42);
    LinearLayer<double> layer(3, 2, rng);
    Tensor<double> out;

    EXPECT_THROW(layer.forward(Tensor<double>({ 2, 4 }), out), InvalidSizeError);
    EXPECT_THROW(layer.forward(Tensor<double>({ 3 }), out), InvalidSizeError);
    EXPECT_THROW(layer.forward(Tensor<double>({ 0, 3 }), out), InvalidSizeError);
}

TEST(LinearLayerTest, BackwardBeforeForwardThrowsInvalidSizeError) {
    RandomEngine rng(42);
    LinearLayer<double> layer(3, 2, rng);
    Tensor<double> dx;

    EXPECT_THROW(layer.backward(Tensor<double>({ 1, 2 }), dx), InvalidSizeError);
}

TEST(LinearLayerTest, BackwardRejectsGradientOfWrongShape) {
    RandomEngine rng(42);
    LinearLayer<double> layer(3, 2, rng);
    Tensor<double> out, dx;
    layer.forward(Tensor<double>({ 4, 3 }, 1.0), out);

    EXPECT_THROW(layer.backward(Tensor<double>({ 3, 2 }), dx), InvalidSizeError);
    EXPECT_THROW(layer.backward(Tensor<double>({ 4, 3 }), dx), InvalidSizeError);
}

TEST(LinearLayerTest, BackwardProducesInputGradientOfInputShape) {
    RandomEngine rng(42);
    LinearLayer<double> layer(3, 2, rng);
    Tensor<double> out, dx;
    layer.forward(Tensor<double>({ 4, 3 }, 1.0), out);

    layer.backward(Tensor<double>({ 4, 2 }, 1.0), dx);

    EXPECT_EQ(dx.shape, (std::vector<std::size_t>{ 4, 3 }));
}

TEST(LinearLayerTest, InputGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    LinearLayer<double> layer(4, 3, rng);
    Tensor<double> x = patternMatrix(5, 4);
    Tensor<double> upstream = patternMatrix(5, 3, 1.0);
    Tensor<double> out, dx;
    layer.forward(x, out);
    layer.zeroGrad();
    layer.backward(upstream, dx);

    auto loss = [&]() {
        Tensor<double> y;
        layer.forward(x, y);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss));
}

TEST(LinearLayerTest, WeightGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    LinearLayer<double> layer(4, 3, rng);
    Tensor<double> x = patternMatrix(5, 4);
    Tensor<double> upstream = patternMatrix(5, 3, 1.0);
    Tensor<double> out, dx;
    layer.forward(x, out);
    layer.zeroGrad();
    layer.backward(upstream, dx);

    auto loss = [&]() {
        Tensor<double> y;
        layer.forward(x, y);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(layer.W.value, layer.W.grad, loss));
}

TEST(LinearLayerTest, BiasGradientIsTheColumnSumOfTheUpstreamGradient) {
    RandomEngine rng(42);
    LinearLayer<double> layer(2, 2, rng);
    Tensor<double> out, dx;
    layer.forward(Tensor<double>({ 3, 2 }, 1.0), out);
    layer.zeroGrad();

    layer.backward(makeMatrix(3, 2, { 1, 10, 2, 20, 3, 30 }), dx);

    EXPECT_DOUBLE_EQ(layer.b.grad[0], 6.0);
    EXPECT_DOUBLE_EQ(layer.b.grad[1], 60.0);
}

TEST(LinearLayerTest, GradientsAccumulateAcrossBackwardCallsUntilZeroGrad) {
    RandomEngine rng(42);
    LinearLayer<double> layer(2, 2, rng);
    Tensor<double> out, dx;
    auto upstream = makeMatrix(1, 2, { 1, 1 });
    layer.forward(Tensor<double>({ 1, 2 }, 1.0), out);
    layer.zeroGrad();

    layer.backward(upstream, dx);
    layer.backward(upstream, dx);
    EXPECT_DOUBLE_EQ(layer.b.grad[0], 2.0);

    layer.zeroGrad();
    EXPECT_DOUBLE_EQ(layer.b.grad[0], 0.0);
    EXPECT_DOUBLE_EQ(layer.W.grad[0], 0.0);
}

TEST(LinearLayerTest, UpdateAppliesTheGradientToWeightsAndBias) {
    RandomEngine rng(42);
    auto layer = makeKnownLayer(rng);
    layer.W.grad[0] = 1.0;
    layer.b.grad[1] = -1.0;

    layer.update(0.1);

    EXPECT_NEAR(layer.W.value[0], 1.0 - 0.1, 1e-12);
    EXPECT_NEAR(layer.b.value[1], -0.5 + 0.1, 1e-12);
    EXPECT_DOUBLE_EQ(layer.W.value[1], 2.0);
}

TEST(LinearLayerTest, LearnsALinearFunctionWithSgd) {
    // Fit y = 2x + 1 with a 1 -> 1 layer and mean-squared-error gradients.
    RandomEngine rng(42);
    LinearLayer<double> layer(1, 1, rng);
    auto x = makeMatrix(4, 1, { 1, 2, 3, 4 });
    std::vector<double> target = { 3, 5, 7, 9 };

    double firstLoss = 0.0, lastLoss = 0.0;
    for (int iteration = 0; iteration < 600; iteration++) {
        Tensor<double> out, dx, dOut({ 4, 1 });
        layer.forward(x, out);
        double loss = 0.0;
        for (std::size_t i = 0; i < 4; i++) {
            double error = out[i] - target[i];
            loss += error * error / 4.0;
            dOut[i] = 2.0 * error / 4.0;
        }
        if (iteration == 0) {
            firstLoss = loss;
        }
        lastLoss = loss;

        layer.zeroGrad();
        layer.backward(dOut, dx);
        layer.update(0.02);
    }

    EXPECT_LT(lastLoss, 0.01 * firstLoss);
    EXPECT_NEAR(layer.W.value[0], 2.0, 0.1);
    EXPECT_NEAR(layer.b.value[0], 1.0, 0.3);
}
