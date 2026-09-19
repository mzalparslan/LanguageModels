#pragma once

#include "Tensor.h"
#include "Validation.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

/**
 * @brief Random number engine used to draw initial weights.
 *
 * There is no global engine. A model creates one from its own seed
 * (default 42, in Debug and Release alike) when it is constructed and passes
 * it down to every layer, so a given seed always reproduces same weights
 * regardless of what else has been constructed.
 */
using RandomEngine = std::mt19937;

/**
 * @brief Weight-update algorithm applied by Parameter::update().
 */
enum class OptimizerKind {
	// Plain gradient descent: param -= lr * clip(grad).
	SGD,
	// Adaptive Moment Estimation; needs a 1-based step count for bias correction.
	Adam
};

/**
 * @brief Which optimizer to apply, plus Adam timestep it needs.
 */
class UpdateRule {
public:
	OptimizerKind kind = OptimizerKind::SGD;
	// 1-based Adam timestep t; unused by SGD.
	std::size_t step = 0;

	static UpdateRule sgd() { return {}; }

	/**
	 * @param step 1-based timestep (bias correction divides by 1 - beta^step,
	 * which is zero at step 0).
	 * @throws InvalidParameterError if step is 0.
	 */
	static UpdateRule adam(std::size_t step) {
		if (step == 0) {
			throw InvalidParameterError("Adam timestep must be >= 1!");
		}
		return { OptimizerKind::Adam, step };
	}
};

/**
 * @brief Learnable parameter: weights/biases plus their gradient and Adam
 * optimizer moment state.
 *
 * Unlike MachineLearningModels::ModelParameters (a plain weight vector +
 * bias for classical linear models), this holds full Tensor-shaped data
 * and extra Adam state (firstMoment, secondMoment) that gradient-based
 * deep learning training needs.
 */
template <typename T>
class Parameter {
public:
	// Weights or biases.
	Tensor<T> value;
	// Gradients (Partial Derivatives) computed during backpropagation.
	Tensor<T> grad;

	// Adaptive Moment Estimation (Adam):
	// It computes individual adaptive learning rates for different parameters
	// based on estimates of first and second moments of gradients.
	// Both are allocated on first Adam update, so SGD-only training
	// never pays for them.
	// 1st moment: exponential moving average of gradients (m in paper).
	Tensor<T> firstMoment;
	// 2nd moment: exponential moving average of squared gradients (v in paper).
	Tensor<T> secondMoment;

	/**
	 * @brief Initialize weights with random variables and gradients with 0.
	 * Discards any Adam moment state from a previous initialization.
	 *
	 * @param rng Engine weights are drawn from; consumed in place, so the
	 * order parameters are initialized in determines their values.
	 * @throws InvalidParameterSizeError If shape is empty or has a zero dimension.
	 * @throws InvalidParameterError If scale is negative.
	 * @throws NaNError, NonFiniteError If scale is not finite.
	 * @see initConstant() for parameters that need no randomness.
	 */
	void init(const std::vector<std::size_t>& shape, T scale, RandomEngine& rng) {
		validation::requireValidShape(shape, "Parameter shape");
		validation::requireNonNegativeFinite(scale, "Parameter initialization scale");

		value = Tensor<T>(shape);
		grad = Tensor<T>(shape, T(0));

		// Initialize Adam Optimizer related parameters later.
		firstMoment = Tensor<T>();
		secondMoment = Tensor<T>();

		// Always randomly init weights to prevent Symmetry Problem.
		// Symmetry Problem: Network would behave like it had only one neuron per layer
		// if weights are initialized with same constant.
		std::normal_distribution<T> dist(T(0), T(1));
		for (auto& x : value.data) {
			x = dist(rng) * scale;
		}
	}

	/**
	 * @brief Initialize every weight to a constant (no randomness) and
	 * gradients to 0. Discards any Adam moment state.
	 *
	 * @param shape Shape of parameter.
	 * @param fill Value given to every weight (e.g. 1 for a normalization scale).
	 * @throws InvalidParameterSizeError If shape is empty or has a zero dimension.
	 * @throws NaNError, NonFiniteError If fill is not finite.
	 */
	void initConstant(const std::vector<std::size_t>& shape, T fill) {
		validation::requireValidShape(shape, "Parameter shape");
		validation::requireFinite(fill, "Parameter constant");

		value = Tensor<T>(shape, fill);
		grad = Tensor<T>(shape, T(0));
		firstMoment = Tensor<T>();
		secondMoment = Tensor<T>();
	}

	/**
	 * @brief Resets gradient to 0. Backward passes accumulate (+=) into
	 * grad, so call this once per training step before them.
	 */
	void zeroGrad() {
		std::fill(grad.data.begin(), grad.data.end(), T(0));
	}

	/**
	 * @brief Update weights with given optimizer.
	 *
	 * @throws InvalidParameterError If lr <= 0.
	 * @throws NaNError, NonFiniteError If lr, a gradient, or an updated weight
	 * is not finite: a bad value is reported before it is stored in the model.
	 * @throws InvalidParameterSizeError If the gradient's size differs from the
	 * weights' size, or the parameter was never initialized.
	 */
	void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
		validation::requirePositiveFinite(lr, "Learning rate");
		validation::requirePositiveSize(value.size(), "Parameter size");
		if (grad.size() != value.size()) {
			throw InvalidParameterSizeError("Parameter gradient size differs from its weights!");
		}

		switch (rule.kind) {
		case OptimizerKind::SGD:
			this->updateNoAdam(lr);
			break;
		case OptimizerKind::Adam:
			this->updateWAdam(lr, rule.step);
			break;
		}
	}

private:
	/**
	 * @brief Element-wise gradient clipping (clip-by-value) to limit
	 * exploding gradients.
	 *
	 * A single huge gradient can throw a weight to inf/NaN in one update.
	 * Clamping each gradient component to [-1, 1] bounds size of any one
	 * update. NaN gradients are not caught (comparisons with NaN are false,
	 * so they pass through unchanged). It does not help with vanishing
	 * gradients.
	 * - Cons -
	 * 1- Distorted Direction: clamping components independently changes the
	 *    direction of gradient vector, altering path of steepest descent.
	 * 2- Slower Convergence: a large gradient is sometimes correct, e.g. when
	 *    it is fixing a massive error.
	 * 3- Masks Deeper Issues: layer norm, bad init or learning rate problems.
	 * Alternative Solution: Global Norm Clipping, which scales whole
	 * gradient (across all parameters) down proportionally if its norm
	 * exceeds a threshold, preserving direction.
	 */
	inline T clipGradient(T grad) {
		if (grad > T(1)) {
			grad = T(1);
		}
		if (grad < T(-1)) {
			grad = T(-1);
		}

		return grad;
	}

	/**
	 * @brief Update weights by using Adam Optimizer.
	 */
	void updateWAdam(T lr, std::size_t adamT, T beta1 = T(0.9), T beta2 = T(0.999), T epsilon = T(1e-8)) {
		if (0 == adamT) {
			throw InvalidParameterError("Adam timestep must be >= 1!");
		}
		validation::requirePositiveFinite(epsilon, "Adam epsilon");
		if (!(beta1 >= T(0) && beta1 < T(1)) || !(beta2 >= T(0) && beta2 < T(1))) {
			throw InvalidParameterError("Adam beta1 and beta2 must be in [0, 1)!");
		}

		// Bias-correction denominators depend only on the timestep, so compute
		// them once; a zero here would divide by zero for every weight.
		auto biasCorrection1 = T(1) - std::pow(beta1, (double)adamT);
		auto biasCorrection2 = T(1) - std::pow(beta2, (double)adamT);
		validation::requireNonZeroDenominator(biasCorrection1, "Adam first-moment bias correction");
		validation::requireNonZeroDenominator(biasCorrection2, "Adam second-moment bias correction");

		// Lazy loading for Adam Optimizer.
		if (firstMoment.size() != value.size()) {
			firstMoment = Tensor<T>(value.shape, T(0));
			secondMoment = Tensor<T>(value.shape, T(0));
		}

		for (std::size_t i = 0; i < value.size(); i++) {
			// clipGradient() lets NaN through unchanged, so reject it first.
			validation::requireFinite(grad[i], "Gradient");
			validation::requireFinite(grad[i], "Gradient");
			T g = clipGradient(grad[i]);
			// 1. Updated biased first moment estimate.
			// m_t = beta1 * m_{t - 1} + (1 - beta1) * g_t
			firstMoment[i] = beta1 * firstMoment[i] + (T(1) - beta1) * g;

			// 2. Update biased second raw moment estimate
			// v_t = beta2 * v_{t - 1} + (1 - beta2) * g_t^2
			secondMoment[i] = beta2 * secondMoment[i] + (T(1) - beta2) * g * g;

			// 3. Compute bias-corrected first moment estimate
			// mHat = m_t / (1 - beta1^adamT)
			T mHat = firstMoment[i] / biasCorrection1;

			// 4. Compute bias-corrected second moment estimate.
			// vHat = v_t / (1 - beta2^adamT)
			T vHat = secondMoment[i] / biasCorrection2;

			// 5. Update parameters
			// theta_t = theta_{t - 1} - lr * mHat / (sqrt(vHat) + epsilon)
			value[i] -= lr * mHat / (std::sqrt(vHat) + epsilon);
			validation::requireFinite(value[i], "Updated weight");
		}
	}

	/**
	 * @brief Update weights without any optimization.
	 */
	void updateNoAdam(T lr) {
		for (std::size_t i = 0; i < value.size(); i++) {
			// clipGradient() would turn an infinite gradient into 1 and pass NaN
			// through, so reject both before clipping.
			validation::requireFinite(grad[i], "Gradient");

			// Descent weights by learning rate * grad.
			value[i] -= lr * clipGradient(grad[i]);
			validation::requireFinite(value[i], "Updated weight");
		}
	}
};
