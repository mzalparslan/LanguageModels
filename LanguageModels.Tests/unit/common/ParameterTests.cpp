#include "pch.h"
#include "Parameter.h"
#include "TestSupport.h"
#include <cmath>
#include <limits>

namespace {
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    const double kInf = std::numeric_limits<double>::infinity();

    // A 1-D parameter with the given weights and gradients.
    Parameter<double> makeParameter(const std::vector<double>& weights,
        const std::vector<double>& gradients) {
        Parameter<double> parameter;
        RandomEngine rng(1);
        parameter.init({ weights.size() }, 1.0, rng);
        for (std::size_t i = 0; i < weights.size(); i++) {
            parameter.value[i] = weights[i];
            parameter.grad[i] = gradients[i];
        }
        return parameter;
    }
}

// --------------------------------------------------------------- UpdateRule

TEST(UpdateRuleTest, DefaultIsSgd) {
    UpdateRule rule = UpdateRule::sgd();

    EXPECT_EQ(rule.kind, OptimizerKind::SGD);
}

TEST(UpdateRuleTest, AdamKeepsItsTimestep) {
    UpdateRule rule = UpdateRule::adam(7);

    EXPECT_EQ(rule.kind, OptimizerKind::Adam);
    EXPECT_EQ(rule.step, 7u);
}

TEST(UpdateRuleTest, AdamTimestepZeroThrowsInvalidParameterError) {
    EXPECT_THROW(UpdateRule::adam(0), InvalidParameterError);
}

// --------------------------------------------------------------------- init

TEST(ParameterTest, InitCreatesWeightsAndZeroGradients) {
    Parameter<double> parameter;
    RandomEngine rng(42);

    parameter.init({ 3, 4 }, 0.5, rng);

    EXPECT_EQ(parameter.value.shape, (std::vector<std::size_t>{ 3, 4 }));
    EXPECT_EQ(parameter.grad.shape, (std::vector<std::size_t>{ 3, 4 }));
    for (std::size_t i = 0; i < parameter.grad.size(); i++) {
        EXPECT_DOUBLE_EQ(parameter.grad[i], 0.0);
    }
}

TEST(ParameterTest, InitWeightsAreRandomAndNotAllEqual) {
    Parameter<double> parameter;
    RandomEngine rng(42);

    parameter.init({ 64 }, 1.0, rng);

    bool anyDifferent = false;
    for (std::size_t i = 1; i < parameter.value.size(); i++) {
        anyDifferent = anyDifferent || parameter.value[i] != parameter.value[0];
    }
    EXPECT_TRUE(anyDifferent);
}

TEST(ParameterTest, InitScaleScalesTheDistribution) {
    Parameter<double> small, large;
    RandomEngine rngA(7), rngB(7);

    small.init({ 200 }, 0.01, rngA);
    large.init({ 200 }, 1.0, rngB);

    // Same seed, so the draws are identical up to the scale factor.
    for (std::size_t i = 0; i < 200; i++) {
        EXPECT_NEAR(small.value[i] * 100.0, large.value[i], 1e-9);
    }
}

TEST(ParameterTest, InitZeroScaleGivesZeroWeights) {
    Parameter<double> parameter;
    RandomEngine rng(42);

    parameter.init({ 5 }, 0.0, rng);

    for (std::size_t i = 0; i < parameter.value.size(); i++) {
        EXPECT_DOUBLE_EQ(parameter.value[i], 0.0);
    }
}

TEST(ParameterTest, SameSeedGivesIdenticalWeights) {
    Parameter<double> first, second;
    RandomEngine rngA(123), rngB(123);

    first.init({ 10 }, 1.0, rngA);
    second.init({ 10 }, 1.0, rngB);

    EXPECT_TRUE(testsupport::tensorsEqual(first.value, second.value));
}

TEST(ParameterTest, DifferentSeedsGiveDifferentWeights) {
    Parameter<double> first, second;
    RandomEngine rngA(1), rngB(2);

    first.init({ 10 }, 1.0, rngA);
    second.init({ 10 }, 1.0, rngB);

    EXPECT_FALSE(testsupport::tensorsEqual(first.value, second.value));
}

TEST(ParameterTest, EngineIsConsumedSoSuccessiveParametersDiffer) {
    Parameter<double> first, second;
    RandomEngine rng(42);

    first.init({ 10 }, 1.0, rng);
    second.init({ 10 }, 1.0, rng);

    EXPECT_FALSE(testsupport::tensorsEqual(first.value, second.value));
}

TEST(ParameterTest, InitRejectsBadShapes) {
    Parameter<double> parameter;
    RandomEngine rng(1);

    EXPECT_THROW(parameter.init({}, 1.0, rng), InvalidParameterSizeError);
    EXPECT_THROW(parameter.init({ 3, 0 }, 1.0, rng), InvalidParameterSizeError);
}

TEST(ParameterTest, InitRejectsBadScale) {
    Parameter<double> parameter;
    RandomEngine rng(1);

    EXPECT_THROW(parameter.init({ 2 }, -0.5, rng), InvalidParameterError);
    EXPECT_THROW(parameter.init({ 2 }, kNaN, rng), NaNError);
    EXPECT_THROW(parameter.init({ 2 }, kInf, rng), NonFiniteError);
}

TEST(ParameterTest, InitDiscardsPreviousAdamMoments) {
    Parameter<double> parameter = makeParameter({ 1.0, 2.0 }, { 0.1, 0.1 });
    parameter.update(0.1, UpdateRule::adam(1));
    ASSERT_EQ(parameter.firstMoment.size(), 2u);

    RandomEngine rng(1);
    parameter.init({ 2 }, 1.0, rng);

    EXPECT_EQ(parameter.firstMoment.size(), 0u);
    EXPECT_EQ(parameter.secondMoment.size(), 0u);
}

// --------------------------------------------------------------- initConstant

TEST(ParameterTest, InitConstantFillsEveryWeight) {
    Parameter<double> parameter;

    parameter.initConstant({ 2, 3 }, 1.0);

    EXPECT_EQ(parameter.value.size(), 6u);
    for (std::size_t i = 0; i < parameter.value.size(); i++) {
        EXPECT_DOUBLE_EQ(parameter.value[i], 1.0);
        EXPECT_DOUBLE_EQ(parameter.grad[i], 0.0);
    }
}

TEST(ParameterTest, InitConstantValidatesShapeAndFill) {
    Parameter<double> parameter;

    EXPECT_THROW(parameter.initConstant({}, 1.0), InvalidParameterSizeError);
    EXPECT_THROW(parameter.initConstant({ 0 }, 1.0), InvalidParameterSizeError);
    EXPECT_THROW(parameter.initConstant({ 2 }, kNaN), NaNError);
    EXPECT_THROW(parameter.initConstant({ 2 }, kInf), NonFiniteError);
}

// ------------------------------------------------------------------- zeroGrad

TEST(ParameterTest, ZeroGradClearsGradientsButKeepsWeights) {
    Parameter<double> parameter = makeParameter({ 1.0, 2.0 }, { 0.5, -0.5 });

    parameter.zeroGrad();

    EXPECT_DOUBLE_EQ(parameter.grad[0], 0.0);
    EXPECT_DOUBLE_EQ(parameter.grad[1], 0.0);
    EXPECT_DOUBLE_EQ(parameter.value[0], 1.0);
    EXPECT_DOUBLE_EQ(parameter.value[1], 2.0);
}

// ------------------------------------------------------------------------ SGD

TEST(ParameterSgdTest, SubtractsLearningRateTimesGradient) {
    Parameter<double> parameter = makeParameter({ 1.0, -2.0 }, { 0.5, -0.25 });

    parameter.update(0.1);

    EXPECT_NEAR(parameter.value[0], 1.0 - 0.1 * 0.5, 1e-12);
    EXPECT_NEAR(parameter.value[1], -2.0 + 0.1 * 0.25, 1e-12);
}

TEST(ParameterSgdTest, ClipsLargeGradientsToOne) {
    Parameter<double> parameter = makeParameter({ 0.0, 0.0 }, { 50.0, -50.0 });

    parameter.update(0.1);

    // Clip-by-value: each component is limited to [-1, 1].
    EXPECT_NEAR(parameter.value[0], -0.1, 1e-12);
    EXPECT_NEAR(parameter.value[1], 0.1, 1e-12);
}

TEST(ParameterSgdTest, DoesNotChangeGradientsOrAllocateAdamMoments) {
    Parameter<double> parameter = makeParameter({ 1.0 }, { 0.3 });

    parameter.update(0.1, UpdateRule::sgd());

    EXPECT_DOUBLE_EQ(parameter.grad[0], 0.3);
    EXPECT_EQ(parameter.firstMoment.size(), 0u);
    EXPECT_EQ(parameter.secondMoment.size(), 0u);
}

TEST(ParameterSgdTest, ZeroGradientLeavesWeightsUnchanged) {
    Parameter<double> parameter = makeParameter({ 1.0, 2.0 }, { 0.0, 0.0 });

    parameter.update(0.5);

    EXPECT_DOUBLE_EQ(parameter.value[0], 1.0);
    EXPECT_DOUBLE_EQ(parameter.value[1], 2.0);
}

// ----------------------------------------------------------------------- Adam

TEST(ParameterAdamTest, FirstStepMovesAgainstTheGradientByAboutLearningRate) {
    // At t = 1 the bias-corrected moments are m_hat = g and v_hat = g^2, so
    // the step is lr * g / (|g| + eps), i.e. ~lr regardless of gradient size.
    Parameter<double> parameter = makeParameter({ 1.0, 1.0 }, { 0.2, -0.9 });

    parameter.update(0.01, UpdateRule::adam(1));

    EXPECT_NEAR(parameter.value[0], 1.0 - 0.01, 1e-6);
    EXPECT_NEAR(parameter.value[1], 1.0 + 0.01, 1e-6);
}

TEST(ParameterAdamTest, AllocatesMomentsLazilyOnFirstUpdate) {
    Parameter<double> parameter = makeParameter({ 1.0, 2.0, 3.0 }, { 0.1, 0.1, 0.1 });
    ASSERT_EQ(parameter.firstMoment.size(), 0u);

    parameter.update(0.01, UpdateRule::adam(1));

    EXPECT_EQ(parameter.firstMoment.size(), 3u);
    EXPECT_EQ(parameter.secondMoment.size(), 3u);
}

TEST(ParameterAdamTest, MomentsFollowTheExponentialMovingAverages) {
    Parameter<double> parameter = makeParameter({ 0.0 }, { 0.5 });

    parameter.update(0.01, UpdateRule::adam(1));

    // m = (1 - beta1) * g, v = (1 - beta2) * g^2
    EXPECT_NEAR(parameter.firstMoment[0], 0.1 * 0.5, 1e-12);
    EXPECT_NEAR(parameter.secondMoment[0], 0.001 * 0.25, 1e-12);
}

TEST(ParameterAdamTest, ClipsGradientBeforeUpdatingMoments) {
    Parameter<double> parameter = makeParameter({ 0.0 }, { 100.0 });

    parameter.update(0.01, UpdateRule::adam(1));

    EXPECT_NEAR(parameter.firstMoment[0], 0.1 * 1.0, 1e-12);
}

TEST(ParameterAdamTest, KeepsMomentumAcrossSteps) {
    Parameter<double> parameter = makeParameter({ 0.0 }, { 1.0 });

    parameter.update(0.01, UpdateRule::adam(1));
    double firstMoment = parameter.firstMoment[0];
    parameter.update(0.01, UpdateRule::adam(2));

    // m2 = beta1 * m1 + (1 - beta1) * g
    EXPECT_NEAR(parameter.firstMoment[0], 0.9 * firstMoment + 0.1, 1e-12);
}

TEST(ParameterAdamTest, ConvergesOnAQuadraticBowl) {
    // Minimize f(w) = (w - 3)^2 with gradient 2 (w - 3).
    Parameter<double> parameter = makeParameter({ 0.0 }, { 0.0 });

    for (std::size_t step = 1; step <= 2000; step++) {
        parameter.grad[0] = 2.0 * (parameter.value[0] - 3.0);
        parameter.update(0.05, UpdateRule::adam(step));
    }

    EXPECT_NEAR(parameter.value[0], 3.0, 1e-2);
}

// ----------------------------------------------------------------- validation

TEST(ParameterUpdateTest, RejectsNonPositiveLearningRate) {
    Parameter<double> parameter = makeParameter({ 1.0 }, { 0.1 });

    EXPECT_THROW(parameter.update(0.0), InvalidParameterError);
    EXPECT_THROW(parameter.update(-0.1), InvalidParameterError);
}

TEST(ParameterUpdateTest, RejectsNonFiniteLearningRate) {
    Parameter<double> parameter = makeParameter({ 1.0 }, { 0.1 });

    EXPECT_THROW(parameter.update(kNaN), NaNError);
    EXPECT_THROW(parameter.update(kInf), NonFiniteError);
}

TEST(ParameterUpdateTest, RejectsUninitializedParameter) {
    Parameter<double> parameter;

    EXPECT_THROW(parameter.update(0.1), InvalidParameterSizeError);
}

TEST(ParameterUpdateTest, RejectsGradientOfDifferentSize) {
    Parameter<double> parameter = makeParameter({ 1.0, 2.0 }, { 0.1, 0.1 });
    parameter.grad = Tensor<double>({ 3 }, 0.1);

    EXPECT_THROW(parameter.update(0.1), InvalidParameterSizeError);
}

TEST(ParameterUpdateTest, RejectsNaNGradientForBothOptimizers) {
    Parameter<double> sgd = makeParameter({ 1.0, 2.0 }, { 0.1, kNaN });
    Parameter<double> adam = makeParameter({ 1.0, 2.0 }, { 0.1, kNaN });

    EXPECT_THROW(sgd.update(0.1, UpdateRule::sgd()), NaNError);
    EXPECT_THROW(adam.update(0.1, UpdateRule::adam(1)), NaNError);
}

TEST(ParameterUpdateTest, RejectsInfiniteGradientForBothOptimizers) {
    // Clipping would silently turn +inf into 1, so it must be rejected first.
    Parameter<double> sgd = makeParameter({ 1.0 }, { kInf });
    Parameter<double> adam = makeParameter({ 1.0 }, { -kInf });

    EXPECT_THROW(sgd.update(0.1, UpdateRule::sgd()), NonFiniteError);
    EXPECT_THROW(adam.update(0.1, UpdateRule::adam(1)), NonFiniteError);
}

TEST(ParameterUpdateTest, RejectsUpdateThatOverflowsAWeight) {
    Parameter<double> parameter = makeParameter({ -1e308 }, { 1.0 });

    EXPECT_THROW(parameter.update(1e308), NonFiniteError);
}
