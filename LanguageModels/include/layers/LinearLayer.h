#pragma once

#include "Parameter.h"
#include "TensorOps.h"

/**
 * @brief Linear Layer (Fully Connected Layer, Dense Layer)
 * implements Affine Transformation which is a linear map
 * followed by a translation:
 * y = xW + b where Linear Part: xW (Matrix Multiplication) that
 * rotates and scales input vector space.
 * and Translation Part: + b (Bias Addition) that shifts origin.
 *
 * @remark Stateful: forward() caches its input, and backward() uses that
 * cache to compute weight gradients, so call backward() right after the
 * forward() it belongs to. Gradients accumulate until zeroGrad().
 *
 * @tparam T Floating-point mode.
 */
template <typename T>
class LinearLayer {
public:
	/**
	 * @brief Weights, shape [In, Out].
	 */
	Parameter<T> W;
	/**
	 * @brief Bias, shape [Out].
	 */
	Parameter<T> b;

	// Input feature count (rows of W).
	std::size_t dIn;
	// Output feature count (columns of W, length of b).
	std::size_t dOut;

	/**
	 * @brief Initialize Linear Layer with given input and output sizes.
	 *
	 * @param inputSize Shape[0] for weights
	 * @param outputSize Shape[1] for weights, Shape[0] for bias
	 * @param rng Engine initial weights are drawn from.
	 * @throws InvalidParameterSizeError If either size is zero.
	 */
	LinearLayer(std::size_t inputSize, std::size_t outputSize,
		RandomEngine& rng)
		: dIn(inputSize), dOut(outputSize)
	{
		// A zero-sized layer has no weights, and He initialization below would
		// divide by dIn.
		validation::requirePositiveSize(inputSize, "LinearLayer input size");
		validation::requirePositiveSize(outputSize, "LinearLayer output size");

		// He Initialization (Kaiming He Initialization) for weights.
		// Helps with convergence when using ReLU activations.
		// It keeps variance of activations and gradients
		// consistent across layers.
		// stddev = sqrt(2 / fan_in)
		// fan_in = number of input units in weight tensor.
		// ReLU kills half neurons (sets negative values to 0). 
		// This halves variance. 
		// To compensate for killing half signal, 
		// we need to double variance of weights.
		W.init({ dIn, dOut }, std::sqrt(T(2.0) / T(dIn)), rng);
		b.init({ dOut }, 0, rng);
	}

	/**
	 * @brief Reset gradients: should be done before each iteration.
	 */
	void zeroGrad() {
		W.zeroGrad();
		b.zeroGrad();
	}

	/**
	 * @brief linear Transformation: Out = x.W + b
	 * x: [Batch * Seq, In]
	 * W: [In, Out]
	 * b: [Out]
	 */
	void forward(const Tensor<T>& x, Tensor<T>& out) {
		validation::requireColumns(x, dIn, "LinearLayer input");
		// Keep input for backward().
		inputCache = x;
		std::size_t batchSize = x.shape[0];

		// Out = xW + b
		MatMul2D(x, W.value, out);

		// Broadcast bias: add b to every row of result.
		for (std::size_t i = 0; i < batchSize; i++) {
			for (std::size_t j = 0; j < dOut; j++) {
				out.data[i * dOut + j] += b.value[j];
			}
		}
	}

	/**
	 * @brief Backward pass: accumulates dW and dB into parameter
	 * gradients and returns gradient with respect to input.
	 *
	 * @param out Gradient of loss w.r.t. forward()'s output
	 * [Batch * Seq, Out] (i.e. dOut).
	 * @param dx Gradient of loss w.r.t. forward()'s input
	 * [Batch * Seq, In].
	 */
	void backward(const Tensor<T>& out, Tensor<T>& dx) {
		if (inputCache.shape.size() != 2) {
			throw InvalidSizeError("LinearLayer::backward() called before forward()!");
		}
		validation::requireShape(out, inputCache.shape[0], dOut, "LinearLayer output gradient");
		std::size_t batchSize = out.shape[0];

		// dB = sum(dOut, dim=0): bias is shared by every row, so its
		// gradient is sum of row gradients.
		for (std::size_t i = 0; i < batchSize; i++) {
			for (std::size_t j = 0; j < dOut; j++) {
				b.grad.data[j] += out.data[i * dOut + j];
			}
		}

		// dW = x^T * dOut, accumulated (+=) so gradients from several
		// backward() calls add up until zeroGrad().
		Tensor<T> xT;
		transpose2D(inputCache, xT);
		Tensor<T> dW;
		MatMul2D(xT, out, dW);

		for (std::size_t i = 0; i < W.grad.size(); i++) {
			W.grad[i] += dW[i];
		}

		// dx = dOut * W^T: gradient handed to previous layer.
		Tensor<T> WT;
		transpose2D(W.value, WT);
		MatMul2D(out, WT, dx);
	}

	/**
	 * @brief Applies accumulated gradients to W and b.
	 *
	 * @param lr Learning rate.
	 * @param rule Optimizer to use (plain SGD by default, or Adam with its
	 * timestep); see UpdateRule.
	 */
	void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
		W.update(lr, rule);
		b.update(lr, rule);
	}

private:
	/**
	 * @brief Copy of last forward() input, needed by backward() to
	 * compute dW = x^T * dOut.
	 */
	Tensor<T> inputCache;
};
