#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <optional>
#include <vector>

#include "CharacterCorpus.h"
#include "DecoderOnlyModel.h"
#include "Exceptions.h"
#include "LogitMetrics.h"
#include "Pipeline.h"
#include "Validation.h"

/**
 * @brief Settings of a character-level GPT and of its training.
 */
template <typename T>
class GptParameters {
public:
	// Number of stacked decoder blocks.
	std::size_t layers = 2;
	// Characters model sees at once; 0 uses longest model's
	// config allows (Config::maxSeqLen).
	std::size_t contextLength = 0;
	// Plain SGD learning rate (the loss is averaged over window).
	T learningRate = T(0.05);
	// Number of updates; each uses next window of training text
	// (wrapping around at its end).
	std::size_t steps = 2000;
	// Report average loss every this many steps; 0 never reports.
	std::size_t logEverySteps = 500;
};

/**
 * @brief Pipeline adapter for a character-level language model with a
 * DecoderOnlyModel (BasicGPT, or any other block type that can be built from
 * a vocabulary size, width, layer count and context length).
 *
 * sample is one character, so training and test data are texts
 * (std::vector<char>). Split them with DataSplitter::sequentialSplit(), which
 * keeps text in order. vocabulary is characters of training
 * text. result compares model with unigram and bigram baselines.
 *
 * ExecutionStrategy is accepted but not used by this model yet.
 */
template <typename T, typename BlockT, typename Config>
	requires std::constructible_from<DecoderOnlyModel<T, BlockT, Config>,
		std::size_t, std::size_t, std::size_t, std::size_t>
class ModelAdapter<T, DecoderOnlyModel<T, BlockT, Config>, GptParameters<T>> {
public:
	using Model = DecoderOnlyModel<T, BlockT, Config>;
	using Sample = char;
	using Parameters = GptParameters<T>;
	using Result = LanguageModelMetrics;

	explicit ModelAdapter(Parameters parameters) : parameters(std::move(parameters)) {}

	/**
	 * @brief Builds vocabulary from text, then trains a new model.
	 *
	 * @throws InvalidSizeError If text is shorter than one context plus
	 * one character, or context is longer than model's config allows.
	 */
	void train(const std::vector<char>& text, ExecutionStrategy /*strategy: not used by this model*/, Logger& logger) {
		context = parameters.contextLength == 0 ? Config::maxSeqLen : parameters.contextLength;
		validation::requirePositiveSize(parameters.layers, "GPT layer count");

		corpus.fit(text, context + 1);
		const std::vector<std::size_t>& ids = corpus.trainingIds();

		logger.info() << "Vocabulary: " << corpus.vocabSize() << " characters, training text: "
			<< ids.size() << " characters, context: " << context << ".";

		model.emplace(corpus.vocabSize(), Config::d_head, parameters.layers, context);

		std::size_t position = 0;
		T lossSum = T(0);
		std::size_t lossCount = 0;

		for (std::size_t step = 1; step <= parameters.steps; step++) {
			// At end of text, wrap to start.
			if (position + context + 1 > ids.size()) {
				position = 0;
			}

			const std::vector<std::size_t> input(ids.begin() + position, ids.begin() + position + context);
			const std::vector<std::size_t> targets(ids.begin() + position + 1, ids.begin() + position + context + 1);
			lossSum += model->trainStep(input, targets, parameters.learningRate);
			lossCount++;
			position += context;

			if (parameters.logEverySteps != 0 && step % parameters.logEverySteps == 0) {
				lastTrainingLoss = static_cast<double>(lossSum) / static_cast<double>(lossCount);
				logger.info() << "Step " << step << " average loss per character: " << lastTrainingLoss;
				lossSum = T(0);
				lossCount = 0;
			}
		}
	}

	/**
	 * @brief Scores model on held-out text, and baselines too.
	 *
	 * text is scored in windows of one context, each independent of the
	 * others. Characters that never occurred in training text count as an
	 * "unknown" character, which model has never learned to predict.
	 *
	 * @throws PipelineStateError If model has not been trained.
	 * @throws InvalidSizeError If text has fewer than two characters.
	 */
	LanguageModelMetrics evaluate(const std::vector<char>& text, Logger& /*logger*/) {
		requireTrained();

		const std::vector<std::size_t> ids = corpus.encode(text);
		validation::requireAtLeast(ids.size(), 2, "Test text length");

		evaluation::MetricsAccumulator accumulator;
		for (std::size_t start = 0; start + 1 < ids.size(); start += context) {
			const std::size_t length = std::min(context, ids.size() - 1 - start);
			const std::vector<std::size_t> input(ids.begin() + start, ids.begin() + start + length);
			const std::vector<std::size_t> targets(ids.begin() + start + 1, ids.begin() + start + length + 1);

			Tensor<T> logits;
			model->forward(input, logits);
			accumulator.addLogits(logits, targets);
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
	Model& trainedModel() {
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
	std::optional<Model> model;
	std::size_t context = 0;
	double lastTrainingLoss = 0.0;

	void requireTrained() const {
		if (!model) {
			throw PipelineStateError("ModelAdapter<DecoderOnlyModel>: model has not been trained.");
		}
	}
};
