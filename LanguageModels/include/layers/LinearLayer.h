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

	// ------------------------------------------------------------------
	// Multi-threaded variants, for layers with a large output (a
	// vocabulary-sized output projection). They give bit-identical results
	// to the methods above, because every output element is still computed
	// with the same operations in the same order; only which thread computes
	// it changes. `threads` is the most threads to use (1 = the caller only).
	// ------------------------------------------------------------------

	/**
	 * @brief zeroGrad() with the large weight gradient cleared by several threads.
	 */
	void zeroGradParallel(std::size_t threads) {
		W.zeroGradParallel(threads);
		b.zeroGrad();
	}

	/**
	 * @brief forward() with the output columns split between threads.
	 *
	 * @see forward
	 */
	void forwardParallel(const Tensor<T>& x, Tensor<T>& out, std::size_t threads) {
		validation::requireColumns(x, dIn, "LinearLayer input");
		// Keep input for backward().
		inputCache = x;
		const std::size_t M = x.shape[0];
		const std::size_t K = dIn;
		const std::size_t N = dOut;

		// Like MatMul2D: reuse out when it already has the right shape.
		if (out.shape.size() == 2 && out.shape[0] == M && out.shape[1] == N) {
			std::fill(out.data.begin(), out.data.end(), T(0));
		}
		else {
			out = Tensor<T>({ M, N }, T(0));
		}

		// One thread owns a range of output columns for every row, so each out[i][j]
		// is built as in MatMul2D: products added in ascending k (zero inputs skipped),
		// then the bias.
		ThreadPool::shared().parallelFor(N, minMultiplyAddsPerThread / std::max<std::size_t>(M * K, 1),
			threads, [&](std::size_t jBegin, std::size_t jEnd) {
				for (std::size_t i = 0; i < M; i++) {
					T* outRow = &out.data[i * N];
					for (std::size_t k = 0; k < K; k++) {
						const T val = x.data[i * K + k];
						if (0 == val) {
							continue;
						}
						const T* weightRow = &W.value.data[k * N];
						for (std::size_t j = jBegin; j < jEnd; j++) {
							outRow[j] += val * weightRow[j];
						}
					}
					for (std::size_t j = jBegin; j < jEnd; j++) {
						outRow[j] += b.value[j];
					}
				}
			});
	}

	/**
	 * @brief backward() split between threads, without the temporary
	 * transposed copies and gradient matrix that backward() builds.
	 *
	 * @see backward
	 */
	void backwardParallel(const Tensor<T>& out, Tensor<T>& dx, std::size_t threads) {
		if (inputCache.shape.size() != 2) {
			throw InvalidSizeError("LinearLayer::backward() called before forward()!");
		}
		validation::requireShape(out, inputCache.shape[0], dOut, "LinearLayer output gradient");
		const std::size_t M = out.shape[0];
		const std::size_t K = dIn;
		const std::size_t N = dOut;

		ThreadPool& pool = ThreadPool::shared();

		// Bias and weight gradients: one thread owns a range of output columns v.
		//   dB[v] += sum over rows i of out[i][v]                  (rows in order)
		//   dW[k][v] += sum over rows i of x[i][k] * out[i][v]     (rows in order,
		//                                                           zero inputs skipped)
		// The sum over rows is finished before it is added to the stored gradient,
		// as backward() does with its temporary dW.
		pool.parallelFor(N, minMultiplyAddsPerThread / std::max<std::size_t>(M * K, 1),
			threads, [&](std::size_t vBegin, std::size_t vEnd) {
				for (std::size_t i = 0; i < M; i++) {
					for (std::size_t v = vBegin; v < vEnd; v++) {
						b.grad.data[v] += out.data[i * N + v];
					}
				}

				std::vector<T> rowSum(vEnd - vBegin);
				for (std::size_t k = 0; k < K; k++) {
					std::fill(rowSum.begin(), rowSum.end(), T(0));
					for (std::size_t i = 0; i < M; i++) {
						const T val = inputCache.data[i * K + k];
						if (0 == val) {
							continue;
						}
						const T* outRow = &out.data[i * N];
						for (std::size_t v = vBegin; v < vEnd; v++) {
							rowSum[v - vBegin] += val * outRow[v];
						}
					}
					T* gradRow = &W.grad.data[k * N];
					for (std::size_t v = vBegin; v < vEnd; v++) {
						gradRow[v] += rowSum[v - vBegin];
					}
				}
			});

		// dx[i][j] = sum over v of out[i][v] * W[j][v] (v in order, zero gradients
		// skipped). Reading W's rows directly avoids transposing the whole matrix.
		if (dx.shape.size() != 2 || dx.shape[0] != M || dx.shape[1] != K) {
			dx = Tensor<T>({ M, K }, T(0));
		}
		pool.parallelFor(M * K, minMultiplyAddsPerThread / std::max<std::size_t>(N, 1),
			threads, [&](std::size_t begin, std::size_t end) {
				for (std::size_t index = begin; index < end; index++) {
					const T* outRow = &out.data[(index / K) * N];
					const T* weightRow = &W.value.data[(index % K) * N];
					T sum = T(0);
					for (std::size_t v = 0; v < N; v++) {
						const T val = outRow[v];
						if (0 == val) {
							continue;
						}
						sum += val * weightRow[v];
					}
					dx.data[index] = sum;
				}
			});
	}

	/**
	 * @brief update() with the weights updated by several threads.
	 */
	void updateParallel(T lr, UpdateRule rule, std::size_t threads) {
		W.updateParallel(lr, rule, threads);
		b.update(lr, rule);
	}

private:
	// Multiply-adds worth giving to one thread: below this, waking a thread
	// costs more than the work it would do.
	static constexpr std::size_t minMultiplyAddsPerThread = 20000;

	/**
	 * @brief Copy of last forward() input, needed by backward() to
	 * compute dW = x^T * dOut.
	 */
	Tensor<T> inputCache;
};
