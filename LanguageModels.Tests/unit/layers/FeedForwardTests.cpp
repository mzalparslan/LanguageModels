#include "pch.h"
#include "FeedForward.h"
#include "TestSupport.h"

using testsupport::patternMatrix;
using testsupport::weightedSum;

TEST(FeedForwardTest, ForwardPreservesRowCountAndModelWidth) {
    RandomEngine rng(42);
    FeedForward<double> ff(4, 16, rng);
    Tensor<double> out;

    ff.forward(patternMatrix(5, 4), out);

    EXPECT_EQ(out.shape, (std::vector<std::size_t>{ 5, 4 }));
}

TEST(FeedForwardTest, ZeroSizesThrowInvalidParameterSizeError) {
    RandomEngine rng(42);

    EXPECT_THROW(FeedForward<double>(0, 8, rng), InvalidParameterSizeError);
    EXPECT_THROW(FeedForward<double>(4, 0, rng), InvalidParameterSizeError);
}

TEST(FeedForwardTest, ForwardRejectsWrongWidth) {
    RandomEngine rng(42);
    FeedForward<double> ff(4, 8, rng);
    Tensor<double> out;

    EXPECT_THROW(ff.forward(patternMatrix(3, 5), out), InvalidSizeError);
}

TEST(FeedForwardTest, ZeroInputGivesZeroOutput) {
    // Biases start at zero and ReLU keeps zeros at zero.
    RandomEngine rng(42);
    FeedForward<double> ff(4, 8, rng);
    Tensor<double> out;

    ff.forward(Tensor<double>({ 2, 4 }, 0.0), out);

    for (std::size_t i = 0; i < out.size(); i++) {
        EXPECT_DOUBLE_EQ(out[i], 0.0);
    }
}

TEST(FeedForwardTest, RowsAreProcessedIndependently) {
    RandomEngine rng(42);
    FeedForward<double> ff(4, 8, rng);
    Tensor<double> x = patternMatrix(3, 4);
    Tensor<double> single({ 1, 4 });
    for (std::size_t j = 0; j < 4; j++) {
        single[j] = x[1 * 4 + j];
    }
    Tensor<double> outAll, outSingle;

    ff.forward(x, outAll);
    ff.forward(single, outSingle);

    for (std::size_t j = 0; j < 4; j++) {
        EXPECT_NEAR(outAll[1 * 4 + j], outSingle[j], 1e-12);
    }
}

TEST(FeedForwardTest, SameSeedGivesIdenticalOutput) {
    RandomEngine rngA(3), rngB(3);
    FeedForward<double> first(4, 8, rngA);
    FeedForward<double> second(4, 8, rngB);
    Tensor<double> outA, outB;
    Tensor<double> x = patternMatrix(3, 4);

    first.forward(x, outA);
    second.forward(x, outB);

    EXPECT_TRUE(testsupport::tensorsEqual(outA, outB));
}

TEST(FeedForwardTest, BackwardBeforeForwardThrowsInvalidSizeError) {
    RandomEngine rng(42);
    FeedForward<double> ff(4, 8, rng);
    Tensor<double> dx;

    EXPECT_THROW(ff.backward(Tensor<double>({ 2, 4 }), dx), InvalidSizeError);
}

TEST(FeedForwardTest, BackwardRejectsGradientWithDifferentRowCount) {
    RandomEngine rng(42);
    FeedForward<double> ff(4, 8, rng);
    Tensor<double> out, dx;
    ff.forward(patternMatrix(3, 4), out);

    EXPECT_THROW(ff.backward(Tensor<double>({ 2, 4 }, 1.0), dx), InvalidSizeError);
    EXPECT_THROW(ff.backward(Tensor<double>(), dx), InvalidSizeError);
}

TEST(FeedForwardTest, InputGradientMatchesFiniteDifferences) {
    RandomEngine rng(42);
    FeedForward<double> ff(4, 8, rng);
    Tensor<double> x = patternMatrix(3, 4);
    Tensor<double> upstream = patternMatrix(3, 4, 1.0);
    Tensor<double> out, dx;
    ff.forward(x, out);
    ff.zeroGrad();
    ff.backward(upstream, dx);

    auto loss = [&]() {
        Tensor<double> y;
        ff.forward(x, y);
        return weightedSum(y, upstream);
    };
    EXPECT_TRUE(testsupport::gradientMatches(x, dx, loss));
}

TEST(FeedForwardTest, TrainingReducesRegressionLoss) {
    RandomEngine rng(42);
    FeedForward<double> ff(4, 16, rng);
    Tensor<double> x = patternMatrix(6, 4);
    Tensor<double> target = patternMatrix(6, 4, 2.0);

    double firstLoss = 0.0, lastLoss = 0.0;
    for (int iteration = 0; iteration < 300; iteration++) {
        Tensor<double> out, dx;
        ff.forward(x, out);
        Tensor<double> dOut(out.shape);
        double loss = 0.0;
        for (std::size_t i = 0; i < out.size(); i++) {
            double error = out[i] - target[i];
            loss += 0.5 * error * error;
            dOut[i] = error;
        }
        if (iteration == 0) {
            firstLoss = loss;
        }
        lastLoss = loss;

        ff.zeroGrad();
        ff.backward(dOut, dx);
        ff.update(0.01);
    }

    EXPECT_LT(lastLoss, 0.5 * firstLoss);
}

TEST(FeedForwardTest, AdamAlsoReducesRegressionLoss) {
    RandomEngine rng(42);
    FeedForward<double> ff(4, 16, rng);
    Tensor<double> x = patternMatrix(6, 4);
    Tensor<double> target = patternMatrix(6, 4, 2.0);

    double firstLoss = 0.0, lastLoss = 0.0;
    for (std::size_t step = 1; step <= 200; step++) {
        Tensor<double> out, dx;
        ff.forward(x, out);
        Tensor<double> dOut(out.shape);
        double loss = 0.0;
        for (std::size_t i = 0; i < out.size(); i++) {
            double error = out[i] - target[i];
            loss += 0.5 * error * error;
            dOut[i] = error;
        }
        if (step == 1) {
            firstLoss = loss;
        }
        lastLoss = loss;

        ff.zeroGrad();
        ff.backward(dOut, dx);
        ff.update(0.01, UpdateRule::adam(step));
    }

    EXPECT_LT(lastLoss, 0.5 * firstLoss);
}
