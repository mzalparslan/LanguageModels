#include "pch.h"
#include "LinearLayer.h"
#include "TestSupport.h"
#include <cmath>

// LinearLayer's *Parallel methods must give bit-identical results to
// forward()/backward()/update(), whatever thread count, so everything is
// compared exactly (no tolerance).

namespace {
    // A wide output, like a vocabulary projection, so work is split into chunks.
    const std::size_t rows = 6;
    const std::size_t inputSize = 32;
    const std::size_t outputSize = 3000;

    // Deterministic values with some exact zeros, to exercise zero-skipping.
    Tensor<double> patterned(std::size_t r, std::size_t c, double phase) {
        Tensor<double> tensor({ r, c });
        for (std::size_t i = 0; i < tensor.size(); i++) {
            tensor[i] = (i % 5 == 0) ? 0.0 : std::sin(0.37 * static_cast<double>(i) + phase);
        }
        return tensor;
    }

    ::testing::AssertionResult sameParameters(const LinearLayer<double>& a, const LinearLayer<double>& b) {
        if (!testsupport::tensorsEqual(a.W.value, b.W.value)) return ::testing::AssertionFailure() << "W differs";
        if (!testsupport::tensorsEqual(a.b.value, b.b.value)) return ::testing::AssertionFailure() << "b differs";
        if (!testsupport::tensorsEqual(a.W.grad, b.W.grad)) return ::testing::AssertionFailure() << "W gradient differs";
        if (!testsupport::tensorsEqual(a.b.grad, b.b.grad)) return ::testing::AssertionFailure() << "b gradient differs";
        return ::testing::AssertionSuccess();
    }
}

TEST(LinearLayerParallelTest, ForwardIsBitIdentical) {
    for (std::size_t threads : { 1u, 2u, 3u, 8u }) {
        RandomEngine rngA(3), rngB(3);
        LinearLayer<double> sequential(inputSize, outputSize, rngA);
        LinearLayer<double> parallel(inputSize, outputSize, rngB);
        Tensor<double> x = patterned(rows, inputSize, 0.1);
        Tensor<double> expected, actual;

        sequential.forward(x, expected);
        parallel.forwardParallel(x, actual, threads);

        EXPECT_TRUE(testsupport::tensorsEqual(expected, actual)) << "threads " << threads;
    }
}

TEST(LinearLayerParallelTest, ForwardReusesAnOutputOfTheRightShape) {
    RandomEngine rng(3);
    LinearLayer<double> layer(inputSize, outputSize, rng);
    Tensor<double> x = patterned(rows, inputSize, 0.1);
    Tensor<double> out;

    layer.forwardParallel(x, out, 4);
    Tensor<double> first = out;
    layer.forwardParallel(x, out, 4); // second call accumulates into a cleared buffer

    EXPECT_TRUE(testsupport::tensorsEqual(first, out));
}

TEST(LinearLayerParallelTest, BackwardIsBitIdentical) {
    for (std::size_t threads : { 1u, 2u, 3u, 8u }) {
        RandomEngine rngA(3), rngB(3);
        LinearLayer<double> sequential(inputSize, outputSize, rngA);
        LinearLayer<double> parallel(inputSize, outputSize, rngB);
        Tensor<double> x = patterned(rows, inputSize, 0.1);
        Tensor<double> dOut = patterned(rows, outputSize, 0.9);
        Tensor<double> out, dxExpected, dxActual;

        sequential.forward(x, out);
        parallel.forwardParallel(x, out, threads);
        sequential.backward(dOut, dxExpected);
        parallel.backwardParallel(dOut, dxActual, threads);

        EXPECT_TRUE(testsupport::tensorsEqual(dxExpected, dxActual)) << "dx, threads " << threads;
        EXPECT_TRUE(sameParameters(sequential, parallel)) << "threads " << threads;
    }
}

TEST(LinearLayerParallelTest, GradientsAccumulateAcrossBackwardCallsLikeTheOriginal) {
    RandomEngine rngA(3), rngB(3);
    LinearLayer<double> sequential(inputSize, outputSize, rngA);
    LinearLayer<double> parallel(inputSize, outputSize, rngB);
    Tensor<double> out, dx1, dx2;

    for (int call = 0; call < 3; call++) {
        Tensor<double> x = patterned(rows, inputSize, 0.1 * call);
        Tensor<double> dOut = patterned(rows, outputSize, 0.5 + call);
        sequential.forward(x, out);
        sequential.backward(dOut, dx1);
        parallel.forwardParallel(x, out, 4);
        parallel.backwardParallel(dOut, dx2, 4);

        ASSERT_TRUE(testsupport::tensorsEqual(dx1, dx2)) << "call " << call;
        ASSERT_TRUE(sameParameters(sequential, parallel)) << "call " << call;
    }
}

TEST(LinearLayerParallelTest, ASmallLayerStillGivesTheSameResult) {
    RandomEngine rngA(3), rngB(3);
    LinearLayer<double> sequential(4, 5, rngA);
    LinearLayer<double> parallel(4, 5, rngB);
    Tensor<double> x = patterned(3, 4, 0.2);
    Tensor<double> dOut = patterned(3, 5, 0.7);
    Tensor<double> outA, outB, dxA, dxB;

    sequential.forward(x, outA);
    parallel.forwardParallel(x, outB, 8);
    sequential.backward(dOut, dxA);
    parallel.backwardParallel(dOut, dxB, 8);

    EXPECT_TRUE(testsupport::tensorsEqual(outA, outB));
    EXPECT_TRUE(testsupport::tensorsEqual(dxA, dxB));
    EXPECT_TRUE(sameParameters(sequential, parallel));
}

TEST(LinearLayerParallelTest, ZeroGradAndUpdateMatchTheSequentialVersions) {
    for (std::size_t threads : { 1u, 4u }) {
        RandomEngine rngA(3), rngB(3);
        LinearLayer<double> sequential(inputSize, outputSize, rngA);
        LinearLayer<double> parallel(inputSize, outputSize, rngB);
        Tensor<double> x = patterned(rows, inputSize, 0.1);
        Tensor<double> dOut = patterned(rows, outputSize, 0.9);
        Tensor<double> out, dxA, dxB;

        for (std::size_t step = 1; step <= 3; step++) {
            sequential.zeroGrad();
            parallel.zeroGradParallel(threads);
            sequential.forward(x, out);
            sequential.backward(dOut, dxA);
            parallel.forwardParallel(x, out, threads);
            parallel.backwardParallel(dOut, dxB, threads);
            sequential.update(0.01, UpdateRule::adam(step));
            parallel.updateParallel(0.01, UpdateRule::adam(step), threads);
        }

        EXPECT_TRUE(sameParameters(sequential, parallel)) << "threads " << threads;
    }
}

TEST(LinearLayerParallelTest, RejectsTheSameInvalidInputAsTheSequentialMethods) {
    RandomEngine rng(3);
    LinearLayer<double> layer(inputSize, outputSize, rng);
    Tensor<double> out, dx;

    EXPECT_THROW(layer.forwardParallel(patterned(rows, inputSize + 1, 0.0), out, 4), InvalidSizeError);
    EXPECT_THROW(layer.forwardParallel(Tensor<double>(), out, 4), InvalidSizeError);
    // backward() needs a forward() first.
    EXPECT_THROW(layer.backwardParallel(patterned(rows, outputSize, 0.0), dx, 4), InvalidSizeError);

    layer.forwardParallel(patterned(rows, inputSize, 0.0), out, 4);
    EXPECT_THROW(layer.backwardParallel(patterned(rows + 1, outputSize, 0.0), dx, 4), InvalidSizeError);
    EXPECT_THROW(layer.backwardParallel(patterned(rows, outputSize - 1, 0.0), dx, 4), InvalidSizeError);
}
