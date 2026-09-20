#include "pch.h"
#include "Parameter.h"
#include "TestSupport.h"
#include <cmath>
#include <limits>

// The *Parallel methods must give bit-identical results to their sequential
// counterparts, whatever the thread count, so these tests compare exactly (no
// tolerance).

namespace {
    const std::size_t bigSize = 40000; // large enough to be split into several chunks

    Parameter<double> makeParameter(std::size_t size, std::uint32_t seed = 7) {
        RandomEngine rng(seed);
        Parameter<double> parameter;
        parameter.init({ size }, 0.5, rng);
        return parameter;
    }

    // Deterministic gradients spanning a wide range, so clipping to [-1, 1] is exercised.
    void fillGradient(Parameter<double>& parameter, double phase) {
        for (std::size_t i = 0; i < parameter.grad.size(); i++) {
            parameter.grad[i] = 3.0 * std::sin(0.013 * static_cast<double>(i) + phase);
        }
    }

    ::testing::AssertionResult sameState(const Parameter<double>& a, const Parameter<double>& b) {
        if (!testsupport::tensorsEqual(a.value, b.value)) {
            return ::testing::AssertionFailure() << "weights differ";
        }
        if (!testsupport::tensorsEqual(a.firstMoment, b.firstMoment)) {
            return ::testing::AssertionFailure() << "first moments differ";
        }
        if (!testsupport::tensorsEqual(a.secondMoment, b.secondMoment)) {
            return ::testing::AssertionFailure() << "second moments differ";
        }
        return ::testing::AssertionSuccess();
    }
}

TEST(ParameterParallelTest, ZeroGradParallelClearsTheWholeGradient) {
    for (std::size_t threads : { 1u, 2u, 8u }) {
        Parameter<double> parameter = makeParameter(bigSize * 20);
        fillGradient(parameter, 0.3);

        parameter.zeroGradParallel(threads);

        for (std::size_t i = 0; i < parameter.grad.size(); i++) {
            ASSERT_EQ(parameter.grad[i], 0.0) << "threads " << threads << ", index " << i;
        }
    }
}

TEST(ParameterParallelTest, SgdUpdateIsBitIdenticalToTheSequentialUpdate) {
    for (std::size_t threads : { 1u, 2u, 3u, 8u }) {
        Parameter<double> sequential = makeParameter(bigSize);
        Parameter<double> parallel = makeParameter(bigSize);

        for (int step = 0; step < 3; step++) {
            fillGradient(sequential, step);
            fillGradient(parallel, step);
            sequential.update(0.05, UpdateRule::sgd());
            parallel.updateParallel(0.05, UpdateRule::sgd(), threads);
        }

        EXPECT_TRUE(sameState(sequential, parallel)) << "threads " << threads;
    }
}

TEST(ParameterParallelTest, AdamUpdateIsBitIdenticalToTheSequentialUpdate) {
    for (std::size_t threads : { 1u, 2u, 3u, 8u }) {
        Parameter<double> sequential = makeParameter(bigSize);
        Parameter<double> parallel = makeParameter(bigSize);

        for (std::size_t step = 1; step <= 5; step++) {
            fillGradient(sequential, static_cast<double>(step));
            fillGradient(parallel, static_cast<double>(step));
            sequential.update(0.01, UpdateRule::adam(step));
            parallel.updateParallel(0.01, UpdateRule::adam(step), threads);
        }

        EXPECT_TRUE(sameState(sequential, parallel)) << "threads " << threads;
        // Adam state really was built and used.
        EXPECT_EQ(parallel.firstMoment.size(), bigSize);
    }
}

TEST(ParameterParallelTest, SmallParametersAreUpdatedCorrectlyToo) {
    Parameter<double> sequential = makeParameter(10);
    Parameter<double> parallel = makeParameter(10);
    fillGradient(sequential, 1.0);
    fillGradient(parallel, 1.0);

    sequential.update(0.1, UpdateRule::adam(1));
    parallel.updateParallel(0.1, UpdateRule::adam(1), 8);

    EXPECT_TRUE(sameState(sequential, parallel));
}

TEST(ParameterParallelTest, SequentialAndParallelUpdatesCanBeMixed) {
    Parameter<double> sequential = makeParameter(bigSize);
    Parameter<double> mixed = makeParameter(bigSize);

    for (std::size_t step = 1; step <= 4; step++) {
        fillGradient(sequential, static_cast<double>(step));
        fillGradient(mixed, static_cast<double>(step));
        sequential.update(0.01, UpdateRule::adam(step));
        if (step % 2 == 0) {
            mixed.updateParallel(0.01, UpdateRule::adam(step), 4);
        }
        else {
            mixed.update(0.01, UpdateRule::adam(step));
        }
    }

    EXPECT_TRUE(sameState(sequential, mixed));
}

TEST(ParameterParallelTest, RejectsTheSameInvalidInputAsUpdate) {
    Parameter<double> parameter = makeParameter(bigSize);
    fillGradient(parameter, 0.0);

    EXPECT_THROW(parameter.updateParallel(0.0, UpdateRule::sgd(), 4), InvalidParameterError);
    EXPECT_THROW(parameter.updateParallel(-1.0, UpdateRule::adam(1), 4), InvalidParameterError);
    EXPECT_THROW(parameter.updateParallel(std::numeric_limits<double>::quiet_NaN(), UpdateRule::sgd(), 4), NaNError);
    // Adam timesteps are 1-based.
    EXPECT_THROW(parameter.updateParallel(0.01, UpdateRule::adam(0), 4), InvalidParameterError);

    Parameter<double> uninitialized;
    EXPECT_THROW(uninitialized.updateParallel(0.01, UpdateRule::sgd(), 4), InvalidParameterSizeError);

    Parameter<double> mismatched = makeParameter(10);
    mismatched.grad = Tensor<double>({ 5 }, 0.0);
    EXPECT_THROW(mismatched.updateParallel(0.01, UpdateRule::sgd(), 4), InvalidParameterSizeError);
}

TEST(ParameterParallelTest, AGradientThatIsNotFiniteAnywhereIsReported) {
    for (std::size_t badIndex : { std::size_t(0), bigSize / 2, bigSize - 1 }) {
        Parameter<double> parameter = makeParameter(bigSize);
        fillGradient(parameter, 0.0);
        parameter.grad[badIndex] = std::numeric_limits<double>::quiet_NaN();

        EXPECT_THROW(parameter.updateParallel(0.01, UpdateRule::sgd(), 8), NaNError) << "index " << badIndex;
        EXPECT_THROW(parameter.updateParallel(0.01, UpdateRule::adam(1), 8), NaNError) << "index " << badIndex;

        parameter.grad[badIndex] = std::numeric_limits<double>::infinity();
        EXPECT_THROW(parameter.updateParallel(0.01, UpdateRule::sgd(), 8), NonFiniteError) << "index " << badIndex;
    }
}
