#pragma once

#include <algorithm>
#include <stdexcept>

#include "LinearLayer.h"

/**
 * @brief Position-wise Feed-Forward Network (ReLU):
 * out = W2 * ReLU(W1 * x + b1) + b2
 *
 * A small neural network built from two Linear layers and an activation
 * function, applied independently to every position. It projects each
 * dModel-wide vector up to dFf, applies ReLU, and projects back to dModel.
 * With dFf = 4 * dModel it holds most of a transformer block's parameters
 * and, for short sequences, most of its FLOPs.
 *
 * @remark Stateful: forward() caches input and hidden activations of its
 * most recent call, and backward() differentiates against that cache. Call
 * backward() right after forward() it belongs to, with a gradient of the
 * same number of rows. Gradients accumulate across backward() calls until
 * zeroGrad().
 */
template <typename T>
class FeedForward {
public:
	/**
	 * @brief Builds both linear layers with freshly initialized weights.
	 *
	 * @param dModel Width of input and output vectors.
	 * @param dFf Width of inner (hidden) layer; commonly 4 * dModel.
	 * @param rng Engine initial weights are drawn from.
	 */
	FeedForward(std::size_t dModel, std::size_t dFf, RandomEngine& rng)
		: expand(dModel, dFf, rng), contract(dFf, dModel, rng) {}

	/**
	 * @brief Clears gradients of both linear layers. Call once per
	 * training step before forward/backward pass, because backward()
	 * accumulates (+=) into them.
	 */
	void zeroGrad() {
		expand.zeroGrad();
		contract.zeroGrad();
	}

	/**
	 * @brief Forward pass: out = contract(ReLU(expand(x))).
	 *
	 * @param x Input [rows, dModel], one row per position.
	 * @param out Output [rows, dModel]; resized as needed.
	 */
	void forward(const Tensor<T>& x, Tensor<T>& out) {
		// Step 1: expand to wider hidden space. [rows, dModel] -> [rows, dFf].
		// This is where most of layer's compute is spent.
		expand.forward(x, hidden);

		// Step 2: ReLU, in place. Negative pre-activations become 0, so
		// `hidden` now holds post-activation values; backward() relies on
		// that to rebuild ReLU mask without storing pre-activations.
		for (auto& v : hidden.data) {
			v = std::max(T(0), v);
		}

		// Step 3: contract back to model width. [rows, dFf] -> [rows, dModel].
		// zeros ReLU produced are skipped by MatMul2D's sparse shortcut.
		contract.forward(hidden, out);
	}

	/**
	 * @brief Backward pass: accumulates weight/bias gradients in both linear
	 * layers and returns gradient with respect to input.
	 *
	 * Applies chain rule through forward()'s steps in reverse order.
	 *
	 * @param dOut Gradient of loss w.r.t. forward()'s output [rows, dModel].
	 * @param dx Gradient of loss w.r.t. forward()'s input [rows, dModel].
	 * @throws InvalidSizeError if forward() has not been called, or dOut
	 * does not have as many rows as last forward() input.
	 */
	void backward(const Tensor<T>& dOut, Tensor<T>& dx) {
		// Guard: backward() reads caches written by forward(); without a
		// matching call shapes below would be empty or mismatched.
		if (hidden.shape.size() != 2) {
			throw InvalidSizeError(
				"FeedForward::backward() called before forward()!");
		}
		if (dOut.shape.empty() || dOut.shape[0] != hidden.shape[0]) {
			throw InvalidSizeError(
				"FeedForward::backward() gradient row count differs from last forward() input!");
		}

		// Step 3 reversed: back through contracting layer. Accumulates
		// its weight/bias gradients and yields gradient at hidden
		// activations. [rows, dModel] -> [rows, dFf].
		Tensor<T> dHidden;
		contract.backward(dOut, dHidden);

		// Step 2 reversed: ReLU derivative is 1 where input was positive
		// and 0 elsewhere, so gradients pass through only units that were
		// active. `hidden` is already post-activation, so hidden <= 0 is
		// equivalent to pre-activation <= 0.
		for (std::size_t i = 0; i < hidden.size(); i++) {
			if (hidden[i] <= T(0)) {
				dHidden[i] = T(0);
			}
		}

		// Step 1 reversed: back through expanding layer. Accumulates its
		// weight/bias gradients and yields dx. [rows, dFf] -> [rows, dModel].
		expand.backward(dHidden, dx);
	}

	/**
	 * @brief Applies accumulated gradients to both linear layers'
	 * weights and biases.
	 *
	 * @param lr Learning rate.
	 * @param rule Optimizer to use (plain SGD by default, or Adam with its
	 * timestep); see UpdateRule.
	 */
	void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
		expand.update(lr, rule);
		contract.update(lr, rule);
	}

private:
	// Expands: dModel -> dFf.
	LinearLayer<T> expand;
	// Contracts: dFf -> dModel.
	LinearLayer<T> contract;
	// Post-ReLU activations of last forward() call.
	Tensor<T> hidden;
};
