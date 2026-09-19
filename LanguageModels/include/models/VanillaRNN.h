#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <cstdint>
#include <cstdlib> 

#include "Metrics.h"
#include "Validation.h"


/**
 * @brief Simple Vanilla Recurrent Neural Network
 *
 * RNNs are designed to process sequential data, such as text, speech and time series.
 * where order of elements is important.
 * Unlike Feed-Forward neural networks, that process inputs independently,
 * RNNs utilize recurrent connections where output of a neuron at one time 
 * step is fed back as input to network at next time step.
 * This enables RNNs to capture temporal dependencies and patterns within sequences.
 *
 * This implementation is character-level: input and output vocabulary are
 * character ids. Each step computes
 * h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h) and predicts next character
 * with softmax(h_t W_hy + b_y). It is trained with backpropagation through
 * time (BPTT) and plain SGD with gradient clipping.
 */
template <typename T = double>
class VanillaRNN {
private:
	// Dimensions:
	// Input and output size at character level.
	std::size_t vocabSize;
	std::size_t hiddenSize;

	// Parameters: Flattened row-major axis data for vectorization.
	// Input to Hidden: vocabSize x hiddenSize
	std::vector<T> rawInputToHidden;
	// Hidden to Hidden: hiddenSize x hiddenSize
	std::vector<T> rawHiddenToHidden;
	// Hidden to Output: hiddenSize x vocabSize
	std::vector<T> rawHiddenToOutput;
	// Hidden bias: hiddenSize
	std::vector<T> biasInHidden;
	// Output bias: vocabSize
	std::vector<T> biasInOutput;

	// Everything forward() computed for one sequence, kept so trainStep() can
	// backpropagate through it without recomputing.
	class Cache {
	public:
		// Hidden states: (T + 1) x hiddenSize (includes hPrev)
		std::vector<std::vector<T>> hiddenStates;
		// Probabilities: T x vocabSize
		std::vector<std::vector<T>> probabilities;
	};

	// Fills weight matrices with small random values (scaled normal
	// draws) and zeroes biases. Small weights keep tanh out of its flat
	// regions at start; random values break symmetry between units.
	// Draws from an engine seeded with randomSeed, so same seed always
	// gives same weights.
	//
	// randomSeed: Seed of engine weights are drawn from.
	// scale: Standard deviation of initial weights.
	void initializeParameters(std::uint32_t randomSeed, T scale = 0.01) {
		std::mt19937 gen(randomSeed);
		
		std::normal_distribution<T> dist(T(0.0), T(1.0));

		auto initMatrix = [&](std::vector<T>& matrix, std::size_t size) {
			matrix.resize(size);

			for (auto& val : matrix) {
				val = dist(gen) * scale;
			}
		};

		initMatrix(this->rawInputToHidden, vocabSize * hiddenSize);
		initMatrix(this->rawHiddenToHidden, hiddenSize * hiddenSize);
		initMatrix(this->rawHiddenToOutput, hiddenSize * vocabSize);

		this->biasInHidden.assign(hiddenSize, T(0));
		this->biasInOutput.assign(vocabSize, T(0));
	}

	// Rejects a character id outside [0, vocabSize). Ids arrive as int, and a
	// negative one would wrap to a huge index and read far outside the weights.
	void requireCharId(int id, const char* what) const {
		if (id < 0) {
			throw InvalidParameterError(std::string(what) + " must not be negative!");
		}
		validation::requireBelow(static_cast<std::size_t>(id), vocabSize, what);
	}

public:
	/**
	 * @param hiddenSize Number of hidden units (size of state vector).
	 * @param vocabSize Number of distinct characters (input and output size).
	 * @param randomSeed Seed for initial weights; same seed always
	 * gives same model.
	 * @throws InvalidParameterSizeError If hiddenSize or vocabSize is zero.
	 */
	VanillaRNN(std::size_t hiddenSize, std::size_t vocabSize, std::uint32_t randomSeed = 42)
		: hiddenSize(hiddenSize), vocabSize(vocabSize) {
		validation::requirePositiveSize(hiddenSize, "VanillaRNN hidden size");
		validation::requirePositiveSize(vocabSize, "VanillaRNN vocabulary size");

		this->initializeParameters(randomSeed);
	}

	/**
	 * @brief Forward pass over one sequence.
	 *
	 * @param inputs Character ids, one per time step (one-hot implicitly).
	 * @param hPrev Hidden state before first step [hiddenSize].
	 * @return per-step hidden states (index 0 is hPrev) and next-character
	 * probabilities [steps x vocabSize].
	 * @throws InvalidSizeError If inputs is empty or hPrev is not [hiddenSize] long.
	 * @throws InvalidParameterError If a character id is outside the vocabulary.
	 * @throws NaNError, NonFiniteError If the state or a probability normalizer is not finite.
	 */
	Cache forward(const std::vector<int> &inputs, const std::vector<T> &hPrev) {
		validation::requireNonEmpty(inputs.size(), "Character sequence");
		validation::requireSameSize(hPrev.size(), hiddenSize, "Hidden state");
		for (T stateValue : hPrev) {
			validation::requireFinite(stateValue, "Hidden state value");
		}

		Cache cache;
		// newHidden[-1]
		cache.hiddenStates.push_back(hPrev);
		std::size_t seqLen = inputs.size();

		// Loop through time steps:
		for (std::size_t t = 0; t < seqLen; t++) {
			std::vector<T> newHidden(this->hiddenSize);
			std::vector<T> newProb(this->vocabSize);

			int charIdx = inputs[t];
			requireCharId(charIdx, "Character id");
			const std::vector<T> & hLast = cache.hiddenStates.back();
			// h_t = std::tanh(W_xh[char_idx] + W_hh * h_{t-1} + b_h)
			for (std::size_t i = 0; i < hiddenSize; i++) {
				// One-hot optimization.
				T val = rawInputToHidden[charIdx * hiddenSize + i];
				// W_hh * h_last
				for (std::size_t k = 0; k < hiddenSize; k++) {
					// rawHiddenToHidden is stored row-major? No.
					// Lets assume row is input and cols is output logic usually.
					val += rawHiddenToHidden[k * hiddenSize + i] * hLast[k];
					// Convention: hNew = hOld * W
					// Lets stick to standard: hNew[i] = Sum(hOld[k] * W[k, i])
				}

				val += this->biasInHidden[i];
				newHidden[i] = std::tanh(val);
			}

			cache.hiddenStates.push_back(newHidden);

			// y = W_hy * newHidden + b_y
			// newProb = softmax(y)
			T sumExp = T(0);
			for (std::size_t i = 0; i < vocabSize; i++) {
				T val = biasInOutput[i];

				for (std::size_t k = 0; k < hiddenSize; k++) {
					val += rawHiddenToOutput[k * vocabSize + i] * newHidden[k];
				}

				// Unnormalized
				newProb[i] = std::exp(val);
				sumExp += newProb[i];
			}

			// A NaN/Inf score or a zero total would divide by zero below.
			validation::requireFinite(sumExp, "Softmax normalizer");
			validation::requireNonZeroDenominator(sumExp, "softmax normalizer");

			// Normalize
			for (std::size_t i = 0; i < vocabSize; i++) {
				newProb[i] /= sumExp;
			}

			cache.probabilities.push_back(newProb);
		}

		return cache;
	}

	/**
	 * @brief Training Step: (Forward + Backward + Update)
	 *
	 * @param inputs Character ids of sequence.
	 * @param targets character that follows each input (same length).
	 * @param hPrev In: hidden state to start from. Out: final hidden
	 * state, so a stateful caller can carry it into next sequence.
	 * @param learningRate SGD step size.
	 * @return Current loss (summed cross-entropy over sequence).
	 * @throws InvalidParameterError If learningRate <= 0, or an id is out of range.
	 * @throws InvalidSizeError If targets and inputs differ in length.
	 * @throws NaNError, NonFiniteError If the loss or a gradient is not finite.
	 */
	T trainStep(const std::vector<int>& inputs,
		const std::vector<int>& targets,
		std::vector<T>& hPrev,
		T learningRate) {
		validation::requirePositiveFinite(learningRate, "Learning rate");
		validation::requireSameSize(targets.size(), inputs.size(), "Target sequence");

		// 1. Forward pass.
		Cache cache = forward(inputs, hPrev);

		// 2. Backward (BPTT: Back Propagation Through Time)
		// Gradients are initialized to zero for hidden state x0.
		std::vector<T> dWXh(rawInputToHidden.size(), 0);
		std::vector<T> dWHh(rawHiddenToHidden.size(), 0);
		std::vector<T> dWHy(rawHiddenToOutput.size(), 0);
		std::vector<T> dBH(biasInHidden.size(), 0);
		std::vector<T> dBY(biasInOutput.size(), 0);

		std::vector<T> dHNext(hiddenSize, 0);

		T loss = 0;
		std::size_t seqLen = inputs.size();

		// Countdown idiom: std::size_t can't go below 0, so decrement happens
		// inside condition (post-decrement) instead of for-loop's
		// increment clause, stopping safely once t reaches 0.
		for (std::size_t t = seqLen; t-- > 0; ) {
			requireCharId(targets[t], "Target id");
			std::size_t target = targets[t];
			std::size_t inputIdx = inputs[t];

			// dy = p - y (Softmax - CrossEntropy Derivative)
			std::vector<T> dy = cache.probabilities[t];
			// dy[y] = p[y] - 1
			dy[target] -= 1;

			loss += -std::log(cache.probabilities[t][target]);

			// Gradients for w_hy and by
			// dy is (1 x vocab), h is (1 x hidden)
			// dW_dy += h.T * dy
			for (std::size_t i = 0; i < vocabSize; i++) {
				dBY[i] += dy[i];

				for (std::size_t k = 0; k < hiddenSize; k++) {
					dWHy[k * vocabSize + i] += cache.hiddenStates[t + 1][k] * dy[i];
				}
			}

			// Gradients for h
			// dh = dy * W_hy.T + dHNext * W_hh.T
			std::vector<T> dh(hiddenSize, 0);
			for (std::size_t k = 0; k < hiddenSize; k++) {
				// From output
				for (std::size_t i = 0; i < vocabSize; i++) {
					dh[k] += dy[i] * rawHiddenToOutput[k * vocabSize + i];
				}
				// From next step hidden.
				for (std::size_t i = 0; i < hiddenSize; i++) {
					dh[k] += dHNext[i] * rawHiddenToHidden[k * hiddenSize + i];
				}
			}

			// Backpropagation through tanh
			// dHRaw = dh * (1 - h^2)
			std::vector<T> dHRaw(hiddenSize);
			for (std::size_t k = 0; k < hiddenSize; k++) {
				T hVal = cache.hiddenStates[t + 1][k];
				dHRaw[k] = dh[k] * (T(1) - hVal * hVal);
			}

			// Gradients for W_xh, W_hh and bh
			for (std::size_t k = 0; k < hiddenSize; k++) {
				dBH[k] += dHRaw[k];

				// dWXh (Sparse input)
				dWXh[inputIdx * hiddenSize + k] += dHRaw[k];

				// dWHh
				// h_prev is cache.hs[t]
				for (std::size_t j = 0; j < hiddenSize; j++) {
					dWHh[j * hiddenSize + k] += cache.hiddenStates[t][j] * dHRaw[k];
				}
			}

			// Pass error to next step (t - 1).
			dHNext = dHRaw;
		}

		// A NaN, or an infinite loss (a target with probability 0), would poison
		// every gradient computed from it.
		validation::requireFinite(loss, "Training loss");

		// 3. Clip Gradients (Exploding gradients mitigation)
		auto clip = [](std::vector<T>& v) {
			for (auto& x : v) {
				x = std::max(T(-5), std::min(x, T(5)));
			}
		};

		clip(dWXh);
		clip(dWHh);
		clip(dWHy);
		clip(dBH);
		clip(dBY);

		// 4. Upgrade parameters (SGD)
		auto update = [&](std::vector<T>& param, const std::vector<T>& grad) {
			for (std::size_t i = 0; i < param.size(); i++) {
				// std::min/std::max let NaN through, so reject it before it reaches a weight.
				validation::requireFinite(grad[i], "Gradient");
				param[i] -= learningRate * grad[i];
			}
		};

		update(this->rawInputToHidden, dWXh);
		update(this->rawHiddenToHidden, dWHh);
		update(this->rawHiddenToOutput, dWHy);
		update(this->biasInHidden, dBH);
		update(this->biasInOutput, dBY);

		// Update hPrev for next sequence 
		// (Stateful RNN usually carries forward, 
		// but for isolated sentences we migh not.
		// Here we assume stateful training if caller passes
		// update h back in next call).
		hPrev = cache.hiddenStates.back();

		return loss;
	}

	/**
	 * @brief Generates characters by repeatedly sampling from model's
	 * next-character distribution and feeding sample back in.
	 *
	 * @param hSeed Hidden state to start from.
	 * @param seedIdx First character id (included in result).
	 * @param n Number of characters to generate after seed.
	 * @return seed followed by n sampled character ids.
	 * @throws InvalidSizeError If hSeed is not [hiddenSize] long.
	 * @throws InvalidParameterError If seedIdx is outside the vocabulary.
	 */
	std::vector<std::size_t> sample(const std::vector<T>& hSeed, std::size_t seedIdx, std::size_t n) {
		validation::requireSameSize(hSeed.size(), hiddenSize, "Hidden state");
		requireCharId(static_cast<int>(seedIdx), "Seed character id");

		std::vector<std::size_t> indices;
		indices.push_back(seedIdx);

		std::vector<T> h = hSeed;
		std::size_t x = seedIdx;

		for (int t = 0; t < n; t++) {
			// Forward one step
			std::vector<T> nextH(hiddenSize);

			for (std::size_t i = 0; i < hiddenSize; i++) {
				T val = this->rawInputToHidden[x * hiddenSize + i];
				for (std::size_t k = 0; k < hiddenSize; k++) {
					val += this->rawHiddenToHidden[k * hiddenSize + i] * h[k];
				}
				val += this->biasInHidden[i];
				nextH[i] = std::tanh(val);
			}

			h = nextH;
			std::vector<T> y(vocabSize);
			T sumExp = T(0);
			for (std::size_t i = 0; i < vocabSize; i++) {
				T val = this->biasInOutput[i];
				for (std::size_t k = 0; k < this->hiddenSize; k++) {
					val += this->rawHiddenToOutput[k * this->vocabSize + i] * h[k];
				}

				y[i] = std::exp(val);
				sumExp += y[i];
			}

			validation::requireFinite(sumExp, "Softmax normalizer");
			validation::requireNonZeroDenominator(sumExp, "softmax normalizer");

			// Sampling (weighted random)
			T r = (T)rand() / (T)RAND_MAX * sumExp;
			std::size_t selected = 0;
			T cumulative = 0;

			for (std::size_t i = 0; i < vocabSize; i++) {
				cumulative += y[i];

				if (r <= cumulative) {
					selected = i;
					break;
				}
			}

			indices.push_back(selected);
			x = selected;
		}

		return indices;
	}

	/**
	 * @brief A fresh all-zero hidden state, for start of a new sequence.
	 */
	std::vector<T> getZeroState() {
		return std::vector<T>(hiddenSize, 0);
	}

	/**
	 * @brief Measures model on a sequence without changing it.
	 *
	 * @param inputs Character ids of sequence.
	 * @param targets character that follows each input.
	 * @param hPrev Hidden state to start from.
	 * @return Loss, perplexity, next-character accuracy and bits per character.
	 * @throws InvalidSizeError If inputs is empty or targets differs in length.
	 * @throws InvalidParameterError If an id is out of range.
	 * @throws NonFiniteError If the loss is infinite (a target with probability 0).
	 */
	Metrics evaluate(const std::vector<int>& inputs, const std::vector<int>& targets, const std::vector<T>& hPrev) {
		validation::requireNonEmpty(inputs.size(), "Character sequence");
		validation::requireSameSize(targets.size(), inputs.size(), "Target sequence");

		Cache cache = forward(inputs, hPrev);

		double totalLoss = 0;
		std::size_t correctPredictions = 0;
		std::size_t seqLen = inputs.size();

		for (std::size_t t = 0; t < seqLen; t++) {
			requireCharId(targets[t], "Target id");
			std::size_t target = targets[t];

			// Loss
			T prob = cache.probabilities[t][target];
			totalLoss += -std::log(prob);

			// Accuracy (argmax)
			std::size_t bestIdx = 0;
			T maxProb = T(-1.0);

			for (std::size_t k = 0; k < vocabSize; k++) {
				if (cache.probabilities[t][k] > maxProb) {
					maxProb = cache.probabilities[t][k];
					bestIdx = k;
				}
			}

			if (bestIdx == target) {
				correctPredictions++;
			}
		}

		return Metrics::fromTotals(totalLoss, correctPredictions, seqLen);
	}
};