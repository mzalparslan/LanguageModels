#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "CharacterCorpus.h"
#include "Exceptions.h"
#include "LogitMetrics.h"
#include "Pipeline.h"
#include "Validation.h"
#include "VanillaRNN.h"

/**
 * @brief Settings of a character-level RNN and of its training.
 */
template <typename T>
class RnnParameters {
public:
	// Width of hidden state.
	std::size_t hiddenSize = 64;
	// Characters unrolled (and backpropagated through) per update.
	std::size_t windowLength = 25;
	// Plain SGD learning rate. loss is summed over a window, so this is
	// much smaller than for a per-character loss.
	T learningRate = T(0.01);
	// Number of updates; each uses next window of training text
	// (wrapping around at its end).
	std::size_t steps = 20000;
	// Report average loss every this many steps; 0 never reports.
	std::size_t logEverySteps = 5000;
	// Characters scored at a time when evaluating.
	std::size_t evaluationWindow = 1000;
	// Seed of model's initial weights.
	std::uint32_t randomSeed = 42;
};

/**
 * @brief Pipeline adapter for a character-level language model with VanillaRNN.
 *
 * sample is one character, so training and test data are texts
 * (std::vector<char>). Split them with DataSplitter::sequentialSplit(), which
 * keeps text in order. vocabulary is characters of training
 * text. result compares model with unigram and bigram baselines.
 *
 * ExecutionStrategy is accepted but not used by this model yet.
 */
template <typename T>
class ModelAdapter<T, VanillaRNN<T>, RnnParameters<T>> {
public:
	using Sample = char;
	using Parameters = RnnParameters<T>;
	using Result = LanguageModelMetrics;

	explicit ModelAdapter(Parameters parameters) : parameters(std::move(parameters)) {}

	/**
	 * @brief Builds vocabulary from text, then trains a new model.
	 *
	 * @throws InvalidSizeError If text is shorter than one window plus one
	 * character, or a size setting is zero.
	 */
	void train(const std::vector<char>& text, ExecutionStrategy /*strategy: not used by this model*/, Logger& logger) {
		validation::requirePositiveSize(parameters.hiddenSize, "RNN hidden size");
		validation::requirePositiveSize(parameters.windowLength, "RNN window length");
		validation::requirePositiveSize(parameters.evaluationWindow, "RNN evaluation window");

		corpus.fit(text, parameters.windowLength + 1);
		const std::vector<std::size_t>& ids = corpus.trainingIds();

		logger.info() << "Vocabulary: " << corpus.vocabSize() << " characters, training text: "
			<< ids.size() << " characters.";

		model.emplace(parameters.hiddenSize, corpus.vocabSize(), parameters.randomSeed);

		const std::size_t window = parameters.windowLength;
		std::vector<T> hidden = model->getZeroState();
		std::size_t position = 0;
		T windowLossSum = T(0);
		std::size_t windowCount = 0;

		for (std::size_t step = 1; step <= parameters.steps; step++) {
			// At end of text, wrap to start with a fresh hidden state.
			if (position + window + 1 > ids.size()) {
				position = 0;
				hidden = model->getZeroState();
			}

			const std::vector<int> inputs = toInts(ids, position, window);
			const std::vector<int> targets = toInts(ids, position + 1, window);
			windowLossSum += model->trainStep(inputs, targets, hidden, parameters.learningRate);
			windowCount++;
			position += window;

			if (parameters.logEverySteps != 0 && step % parameters.logEverySteps == 0) {
				lastTrainingLoss = static_cast<double>(windowLossSum)
					/ static_cast<double>(windowCount * window);
				logger.info() << "Step " << step << " average loss per character: " << lastTrainingLoss;
				windowLossSum = T(0);
				windowCount = 0;
			}
		}
	}

	/**
	 * @brief Scores model on held-out text, and baselines too.
	 *
	 * Characters that never occurred in training text count as an
	 * "unknown" character, which model has never learned to predict.
	 *
	 * @throws PipelineStateError If model has not been trained.
	 * @throws InvalidSizeError If text has fewer than two characters.
	 */
	LanguageModelMetrics evaluate(const std::vector<char>& text, Logger& /*logger*/) {
		requireTrained();

		const std::vector<std::size_t> ids = corpus.encode(text);
		validation::requireAtLeast(ids.size(), 2, "Test text length");

		// One window at a time, each from a fresh hidden state; totals are
		// pooled so result is same as scoring whole text at once
		// (up to state reset at window boundaries).
		evaluation::MetricsAccumulator accumulator;
		for (std::size_t start = 0; start + 1 < ids.size(); start += parameters.evaluationWindow) {
			const std::size_t length = std::min(parameters.evaluationWindow, ids.size() - 1 - start);
			const Metrics part = model->evaluate(toInts(ids, start, length),
				toInts(ids, start + 1, length), model->getZeroState());

			const double count = static_cast<double>(length);
			accumulator.add(part.loss * count,
				static_cast<std::size_t>(std::llround(part.accuracy * count)), length);
		}

		LanguageModelMetrics result = corpus.scoreBaselines(ids);
		result.model = accumulator.result();
		return result;
	}

	/**
	 * @brief trained model.
	 *
	 * @throws PipelineStateError If model has not been trained.
	 */
	VanillaRNN<T>& trainedModel() {
		requireTrained();
		return *model;
	}

	/**
	 * @brief Average loss per character over last logged stretch of training.
	 */
	double finalTrainingLoss() const { return lastTrainingLoss; }

private:
	Parameters parameters;
	CharacterCorpus corpus;
	std::optional<VanillaRNN<T>> model;
	double lastTrainingLoss = 0.0;

	void requireTrained() const {
		if (!model) {
			throw PipelineStateError("ModelAdapter<VanillaRNN>: model has not been trained.");
		}
	}

	// VanillaRNN takes character ids as int.
	static std::vector<int> toInts(const std::vector<std::size_t>& ids, std::size_t begin, std::size_t length) {
		std::vector<int> result;
		result.reserve(length);
		for (std::size_t i = begin; i < begin + length; i++) {
			result.push_back(static_cast<int>(ids[i]));
		}
		return result;
	}
};
