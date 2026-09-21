#pragma once

// The kernels of cuda::VocabularyHead. Each one handles one part of the training step.
// They are templates for float and double, so they have to be compiled in the same file
// that launches them (CudaVocabularyHead.cu) unless relocatable device code is used.
//
// Layout of the work: every kernel handles one part of the training step, and each
// keeps the order of the arithmetic of the CPU code where that is cheap to do:
//  - output scores: one thread per (position, word), summing over the width in order;
//  - softmax + cross-entropy: one block per position, a parallel max and sum;
//  - projection gradient: one thread per (width index, word), summing over positions in order;
//  - decoder-output gradient: one block per (position, width index), a parallel sum over words;
//  - embedding gradient: a single block, positions added in order (so a word that occurs
//    twice gets the same result every time, and no atomic additions are needed);
//  - optimizer: one thread per weight.

#include "DeviceMath.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda {

	namespace detail {

		// out[r] = table[ids[r]], one row per id.
		template <typename T>
		__global__ void gatherRowsKernel(const T* table, const int* ids, T* out, std::size_t rows, std::size_t width) {
			const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			if (i < rows * width) {
				out[i] = table[static_cast<std::size_t>(ids[i / width]) * width + i % width];
			}
		}

		// Adds rowGradients[r] into grad[ids[r]] for r in order. One block; each thread
		// owns some columns, so every element is added in position order.
		template <typename T>
		__global__ void scatterAddRowsKernel(T* grad, const int* ids, const T* rowGradients, std::size_t rows, std::size_t width) {
			for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
				for (std::size_t r = 0; r < rows; r++) {
					grad[static_cast<std::size_t>(ids[r]) * width + column] += rowGradients[r * width + column];
				}
			}
		}

		// logits[r][v] = sum over k of x[r][k] * W[k][v], plus bias[v]. Grid: x over words, y over positions.
		template <typename T>
		__global__ void projectKernel(const T* x, const T* weights, const T* bias, T* logits,
			std::size_t width, std::size_t vocab) {
			const std::size_t v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			const std::size_t r = blockIdx.y;
			if (v < vocab) {
				T sum = T(0);
				for (std::size_t k = 0; k < width; k++) {
					sum += x[r * width + k] * weights[k * vocab + v];
				}
				logits[r * vocab + v] = sum + bias[v];
			}
		}

		// One block per position. Turns the row of logits into (probabilities - one-hot label),
		// which is the gradient of the cross-entropy loss with respect to the logits, and
		// records -log(probability of the label).
		template <typename T>
		__global__ void softmaxCrossEntropyKernel(T* logits, const int* labels, T* rowLoss, std::size_t vocab) {
			__shared__ T shared[32];
			T* row = logits + static_cast<std::size_t>(blockIdx.x) * vocab;

			T maxValue = T(-1e30);
			for (std::size_t v = threadIdx.x; v < vocab; v += blockDim.x) {
				maxValue = fmax(maxValue, row[v]);
			}
			maxValue = blockMax(maxValue, shared);

			T sum = T(0);
			for (std::size_t v = threadIdx.x; v < vocab; v += blockDim.x) {
				const T e = exp(row[v] - maxValue);
				row[v] = e;
				sum += e;
			}
			sum = blockSum(sum, shared);

			for (std::size_t v = threadIdx.x; v < vocab; v += blockDim.x) {
				row[v] /= sum;
			}
			__syncthreads();

			if (threadIdx.x == 0) {
				const int label = labels[blockIdx.x];
				const T probability = row[label];
				rowLoss[blockIdx.x] = -log(probability);
				row[label] = probability - T(1);
			}
		}

		// gradWeights[k][v] += sum over positions r of x[r][k] * dlogits[r][v];
		// gradBias[v] += sum over r of dlogits[r][v]. Grid: x over words, y over width.
		template <typename T>
		__global__ void projectionGradientKernel(const T* x, const T* dlogits, T* gradWeights, T* gradBias,
			std::size_t rows, std::size_t width, std::size_t vocab) {
			const std::size_t v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			const std::size_t k = blockIdx.y;
			if (v < vocab) {
				T sum = T(0);
				for (std::size_t r = 0; r < rows; r++) {
					sum += x[r * width + k] * dlogits[r * vocab + v];
				}
				gradWeights[k * vocab + v] += sum;

				if (k == 0) {
					T biasSum = T(0);
					for (std::size_t r = 0; r < rows; r++) {
						biasSum += dlogits[r * vocab + v];
					}
					gradBias[v] += biasSum;
				}
			}
		}

		// dx[r][k] = sum over words v of dlogits[r][v] * W[k][v]. One block per (r, k).
		template <typename T>
		__global__ void inputGradientKernel(const T* dlogits, const T* weights, T* dx, std::size_t width, std::size_t vocab) {
			__shared__ T shared[32];
			const std::size_t r = blockIdx.x / width;
			const std::size_t k = blockIdx.x % width;

			T sum = T(0);
			for (std::size_t v = threadIdx.x; v < vocab; v += blockDim.x) {
				sum += dlogits[r * vocab + v] * weights[k * vocab + v];
			}
			sum = blockSum(sum, shared);
			if (threadIdx.x == 0) {
				dx[blockIdx.x] = sum;
			}
		}

		// value -= lr * clip(gradient). Reports values that are not finite through the flag.
		template <typename T>
		__global__ void sgdKernel(T* value, const T* gradient, std::size_t count, T learningRate, int* flag) {
			const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			if (i < count) {
				const T g = gradient[i];
				flagIfNotFinite(g, flag);
				const T updated = value[i] - learningRate * clipToUnit(g);
				flagIfNotFinite(updated, flag);
				value[i] = updated;
			}
		}

		// One Adam step, with the same arithmetic as Parameter::updateWAdamRange().
		template <typename T>
		__global__ void adamKernel(T* value, const T* gradient, T* first, T* second, std::size_t count,
			T learningRate, T beta1, T beta2, T epsilon, T biasCorrection1, T biasCorrection2, int* flag) {
			const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			if (i < count) {
				T g = gradient[i];
				flagIfNotFinite(g, flag);
				g = clipToUnit(g);

				first[i] = beta1 * first[i] + (T(1) - beta1) * g;
				second[i] = beta2 * second[i] + (T(1) - beta2) * g * g;

				const T firstHat = first[i] / biasCorrection1;
				const T secondHat = second[i] / biasCorrection2;
				const T updated = value[i] - learningRate * firstHat / (sqrt(secondHat) + epsilon);
				flagIfNotFinite(updated, flag);
				value[i] = updated;
			}
		}

	} // namespace detail

} // namespace cuda
