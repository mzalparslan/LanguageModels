#include "pch.h"
#include "RMSNorm.h"
#include "TestSupport.h"
#include <cmath>
#include <limits>

using testsupport::makeMatrix;
using testsupport::patternMatrix;
using testsupport::tensorsNear;
using testsupport::weightedSum;

TEST(RMSNormTest, InitialScaleIsOne) {
    RMSNorm<double> norm(4);

    EXPECT_EQ(norm.weight.value.size(), 4u);
    for (std::size_t i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(norm.weight.value[i], 1.0);
    }
}

TEST(RMSNormTest, InvalidConstructionThrows) {
    EXPECT_THROW(RMSNorm<double>(0), InvalidParameterSizeError);
    EXPECT_THROW(RMSNorm<double>(4, 0.0), InvalidParameterError);
    EXPECT_THROW(RMSNorm<double>(4, -1e-5), InvalidParameterError);
    EXPECT_THROW(RMSNorm<double>(4, std::numeric_limits<double>::quiet_NaN()), NaNError);
    EXPECT_THROW(RMSNorm<double>(4, std::numeric_limits<double>::infinity()), NonFiniteError);
}

TEST(RMSNormTest, ForwardDividesByRootMeanSquare) {
    RMSNorm<double> norm(2, 1e-5);
    auto x = makeMatrix(1, 2, { 3, 4 });
    Tensor<double> out;

    norm.forward(x, out);

    double rms = std::sqrt((9.0 + 16.0) / 2.0 + 1e-5);
    ASSERT_EQ(out.shape, (std::vector<std::size_t>{ 1, 2 }));
    EXPECT_NEAR(out[0], 3.0 / rms, 1e-12);
    EXPECT_NEAR(out[1], 4.0 / rms, 1e-12);
}

TEST(RMSNormTest, OutputRowsHaveUnitRootMeanSquare) {
    RMSNorm<double> norm(6);
    auto x = patternMatrix(4, 6);
    Tensor<double> out;

    norm.forward(x, out);

    for (std::size_t row = 0; row < 4; row++) {
        double sumSquares = 0.0;
        for (std::size_t j = 0; j < 6; j++) {
            sumSquares += out[row * 6 + j] * out[row * 6 + j];
        }
        EXPECT_NEAR(std::sqrt(sumSquares / 6.0), 1.0, 1e-3);
    }
}

TEST(RMSNormTest, ScaleMultipliesEachFeature) {
    RMSNorm<double> norm(2);
    Tensor<double> x = makeMatrix(1, 2, { 3, 4 });
    Tensor<double> unscaled, scaled;
    norm.forward(x, unscaled);

    norm.weight.value[0] = 2.0;
    norm.weight.value[1] = -1.0;
    norm.forward(x, scaled);

    EXPECT_NEAR(scaled[0], 2.0 * unscaled[0], 1e-12);
    EXPECT_NEAR(scaled[1], -1.0 * unscaled[1], 1e-12);
}

TEST(RMSNormTest, RowsAreNormalizedIndependently) {
    RMSNorm<double> norm(2);
    Tensor<double> both = makeMatrix(2, 2, { 3, 4, 30, 40 });
    Tensor<double> out;

    norm.forward(both, out);

    // A row that is 10x larger normalizes to (almost exactly) same values.
    EXPECT_NEAR(out[0], out[2], 1e-4);
    EXPECT_NEAR(out[1], out[3], 1e-4);
}

TEST(RMSNormTest, ZeroInputGivesZeroOutputWithoutDividingByZero) {
    RMSNorm<double> norm(3);
    Tensor<double> x({ 2, 3 }, 0.0);
    Tensor<double> out;

    EXPECT_NO_THROW(norm.forward(x, out));
    for (std::size_t i = 0; i < out.size(); i++) {
        EXPECT_DOUBLE_EQ(out[i], 0.0);
    }
}

TEST(RMSNormTest, NonFiniteInputThrows) {
    RMSNorm<double> norm(2);
    Tensor<double> out;

    EXPECT_THROW(norm.forward(makeMatrix(1, 2, { 1, std::numeric_limits<double>::quiet_NaN() }), out), NonFiniteError);
    EXPECT_THROW(norm.forward(makeMatrix(1, 2, { 1, std::numeric_limits<double>::infinity() }), out), NonFiniteError);
}

TEST(RMSNormTest, ForwardRejectsWrongWidthAndNonMatrix) {
    RMSNorm<double> norm(4);
    Tensor<double> out;

    EXPECT_THROW(norm.forward(Tensor<double>({ 2, 3 }), out), InvalidSizeError);
    EXPECT_THROW(norm.forward(Tensor<double>({ 4 }), out), InvalidSizeError);
}

TEST(RMSNormTest, BackwardBeforeForwardThrowsInvalidSizeError) {
    RMSNorm<double> norm(2);
    Tensor<double> dx;

    EXPECT_THROW(norm.backward(makeMatrix(1, 2, { 1, 1 }), dx), InvalidSizeError);
}

TEST(RMSNormTest, BackwardRejectsGradientOfWrongShape) {
    RMSNorm<double> norm(2);
    Tensor<double> out, dx;
    norm.forward(makeMatrix(2, 2, { 1, 2, 3, 4 }), out);

    EXPECT_THROW(norm.backward(makeMatrix(1, 2, { 1, 1 }), dx), InvalidSizeError);
    EXPECT_THROW(norm.backward(makeMatrix(2, 3, { 1, 1, 1, 1, 1, 1 }), dx), InvalidSizeError);
}

TEST(RMSNormTest, InputGradientMatchesFiniteDifferences) {
    RMSNorm<double> norm(5);
    norm.weight.value = patternMatrix(1, 5, 0.3);
    norm.weight.value.shape = { 5 };
    Tensor<double> x = patternMatrix(3, 5, 1.0);
    Tensor<double> upstream = patternMatrix(3, 5, 2.0);
    Tensor<double> out, dx;
    norm.forward(x, out);

    norm.zeroGrad();
    norm.backward(upstream, dx);

    auto loss = [&]() {
        Tensor<double> y;
        norm.forward(x, y);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss));
}

TEST(RMSNormTest, ScaleGradientMatchesFiniteDifferences) {
    RMSNorm<double> norm(5);
    Tensor<double> x = patternMatrix(3, 5, 1.0);
    Tensor<double> upstream = patternMatrix(3, 5, 2.0);
    Tensor<double> out, dx;
    norm.forward(x, out);

    norm.zeroGrad();
    norm.backward(upstream, dx);

    auto loss = [&]() {
        Tensor<double> y;
        norm.forward(x, y);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(norm.weight.value, norm.weight.grad, loss));
}

TEST(RMSNormTest, ScaleGradientAccumulatesUntilZeroGrad) {
    RMSNorm<double> norm(2);
    auto x = makeMatrix(1, 2, { 1, 2 });
    auto upstream = makeMatrix(1, 2, { 1, 1 });
    Tensor<double> out, dx;
    norm.forward(x, out);

    norm.backward(upstream, dx);
    double once = norm.weight.grad[0];
    norm.backward(upstream, dx);

    EXPECT_NEAR(norm.weight.grad[0], 2.0 * once, 1e-12);
    norm.zeroGrad();
    EXPECT_DOUBLE_EQ(norm.weight.grad[0], 0.0);
}

TEST(RMSNormTest, UpdateMovesScaleAlongNegativeGradient) {
    RMSNorm<double> norm(2);
    norm.weight.grad[0] = 0.5;
    norm.weight.grad[1] = -0.5;

    norm.update(0.1);

    EXPECT_NEAR(norm.weight.value[0], 1.0 - 0.05, 1e-12);
    EXPECT_NEAR(norm.weight.value[1], 1.0 + 0.05, 1e-12);
}
