#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "BERT.h"
#include "ConfusionMatrix.h"
#include "Exceptions.h"
#include "LogitMetrics.h"
#include "Metrics.h"
#include "Pipeline.h"
#include "Validation.h"
#include "WordTokenizer.h"

/**
 * @brief Settings of a BERT model and of its training.
 */
template <typename T>
class BertParameters {
public:
	// Model width.
	std::size_t modelWidth = 32;
	// Number of encoder layers.
	std::size_t layers = 2;
	// Longest input in tokens, [CLS] and [SEP] included; longer sentences are cut.
	std::size_t maxLength = 64;
	// Adam learning rate.
	T learningRate = T(0.001);
	// Passes over training sentences (one optimizer step per sentence).
	std::size_t epochs = 3;
	// Share of words hidden by [MASK] in every example (at least one).
	double maskFraction = 0.15;
	// Largest word vocabulary, special tokens included; rarer words become [UNK].
	std::size_t maxVocabSize = 2000;
	// Report average loss every this many epochs; 0 never reports.
	std::size_t logEveryEpochs = 1;
	// Seed of model's initial weights.
	std::uint32_t randomSeed = 42;
	// Seed of random choices that build examples: which sentence follows
	// which, and which words are masked.
	std::uint32_t dataSeed = 42;
};

/**
 * @brief How well a BERT model does on held-out sentences.
 */
class BertMetrics {
public:
	// Loss, perplexity and accuracy of guessing hidden words.
	Metrics maskedLanguageModel;
	// Next-sentence prediction: rows are truth (0 = IsNext, 1 = NotNext),
	// columns what model said. Also gives accuracy, precision and recall.
	ConfusionMatrix nextSentence{ 2 };
	// Number of examples scored.
	std::size_t examples = 0;
};

/**
 * @brief Pipeline adapter for BERT (masked words and next-sentence prediction).
 *
 * sample is one sentence, so training and test data are lists of
 * sentences in reading order (DataLoader::loadLines(); split them with
 * DataSplitter::sequentialSplit() to keep order, which "next sentence"
 * relies on). Every training step builds one example from a sentence: it
 * pairs it with sentence that really follows or with a random one, and
 * hides some of words. Evaluation builds its examples same way from
 * test sentences, with a fixed seed so results are repeatable.
 *
 * ExecutionStrategy is accepted but not used by this model yet.
 */
template <typename T>
class ModelAdapter<T, BertModel<T>, BertParameters<T>> {
public:
	using Sample = std::string;
	using Parameters = BertParameters<T>;
	using Result = BertMetrics;

	explicit ModelAdapter(Parameters parameters)
		: parameters(std::move(parameters)),
		tokenizer({ "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]" }, "[UNK]") {}

	/**
	 * @brief Builds vocabulary from sentences, then trains a new model.
	 *
	 * @throws InvalidSizeError If fewer than two sentences have a word, or
	 * maxLength is too short to hold two one-word sentences.
	 */
	void train(const std::vector<std::string>& sentences, ExecutionStrategy /*strategy: not used by this model*/, Logger& logger) {
		// [CLS] a [SEP] b [SEP] is five tokens.
		validation::requireAtLeast(parameters.maxLength, 5, "BERT maximum length");
		validation::requirePositiveSize(parameters.epochs, "BERT epoch count");
		validateMaskFraction();

		tokenizer.fit(sentences, parameters.maxVocabSize);
		const std::vector<std::vector<std::size_t>> encoded = encodeSentences(sentences);
		validation::requireAtLeast(encoded.size(), 2, "Training sentences with words");
		if (encoded.size() < sentences.size()) {
			logger.warning() << (sentences.size() - encoded.size()) << " training sentences skipped (no words).";
		}

		logger.info() << "Vocabulary: " << tokenizer.size() << " words, training sentences: " << encoded.size();

		model.emplace(tokenizer.size(), parameters.modelWidth, parameters.layers,
			parameters.maxLength, parameters.randomSeed);

		std::mt19937 random(parameters.dataSeed);
		std::size_t adamStep = 1; // Adam timestep: 1-based, increased after every example
		for (std::size_t epoch = 0; epoch < parameters.epochs; epoch++) {
			T totalLoss = T(0);
			std::size_t used = 0;
			for (std::size_t i = 0; i < encoded.size(); i++) {
				const Example example = makeExample(encoded, i, random);
				// A pair made only of [UNK] words has nothing to hide.
				if (!hasMaskedWord(example)) {
					continue;
				}
				totalLoss += model->trainStep(example.inputIds, example.typeIds, example.mlmLabels,
					example.nspLabel, parameters.learningRate, UpdateRule::adam(adamStep));
				adamStep++;
				used++;
			}
			validation::requireNonEmpty(used, "Training examples with a maskable word");

			lastTrainingLoss = static_cast<double>(totalLoss) / static_cast<double>(used);
			if (parameters.logEveryEpochs != 0 && epoch % parameters.logEveryEpochs == 0) {
				logger.info() << "Epoch " << epoch << " average loss per example: " << lastTrainingLoss;
			}
		}
	}

	/**
	 * @brief Scores model on held-out sentences.
	 *
	 * @throws PipelineStateError If model has not been trained.
	 * @throws InvalidSizeError If fewer than two sentences have a word.
	 */
	BertMetrics evaluate(const std::vector<std::string>& sentences, Logger& /*logger*/) {
		requireTrained();

		const std::vector<std::vector<std::size_t>> encoded = encodeSentences(sentences);
		validation::requireAtLeast(encoded.size(), 2, "Test sentences with words");

		std::mt19937 random(parameters.dataSeed);
		evaluation::MetricsAccumulator maskedWords;
		BertMetrics result;

		for (std::size_t i = 0; i < encoded.size(); i++) {
			const Example example = makeExample(encoded, i, random);

			// Masked words: score only positions that were hidden.
			Tensor<T> logits;
			model->predictMaskedLogits(example.inputIds, example.typeIds, logits);
			const std::size_t vocab = logits.shape[1];
			std::vector<std::size_t> hiddenLabels;
			std::vector<T> hiddenRows;
			for (std::size_t position = 0; position < example.mlmLabels.size(); position++) {
				if (example.mlmLabels[position] != 0) {
					hiddenLabels.push_back(example.mlmLabels[position]);
					hiddenRows.insert(hiddenRows.end(), logits.data.begin() + position * vocab,
						logits.data.begin() + (position + 1) * vocab);
				}
			}
			if (!hiddenLabels.empty()) {
				Tensor<T> hidden({ hiddenLabels.size(), vocab });
				hidden.data = std::move(hiddenRows);
				maskedWords.addLogits(hidden, hiddenLabels);
			}

			// Next sentence: which of two scores is higher.
			Tensor<T> nspLogits;
			model->predictNextSentenceLogits(example.inputIds, example.typeIds, nspLogits);
			result.nextSentence.add(example.nspLabel,
				nspLogits.data[1] > nspLogits.data[0] ? std::size_t(1) : std::size_t(0));
		}

		validation::requireNonEmpty(maskedWords.count(), "Test words that could be masked");
		result.maskedLanguageModel = maskedWords.result();
		result.examples = encoded.size();
		return result;
	}

	/**
	 * @brief trained model.
	 *
	 * @throws PipelineStateError If model has not been trained.
	 */
	BertModel<T>& trainedModel() {
		requireTrained();
		return *model;
	}

	/**
	 * @brief vocabulary built from training sentences.
	 */
	const WordTokenizer& wordTokenizer() const { return tokenizer; }

	/**
	 * @brief Average loss per example over last epoch trained.
	 */
	double finalTrainingLoss() const { return lastTrainingLoss; }

private:
	// One BERT input with its two targets.
	class Example {
	public:
		std::vector<std::size_t> inputIds;  // [CLS] A [SEP] B [SEP], some words replaced by [MASK]
		std::vector<std::size_t> typeIds;   // 0 for [CLS] A [SEP], 1 for B [SEP]
		std::vector<std::size_t> mlmLabels; // original word at masked positions, 0 elsewhere
		std::size_t nspLabel = 0;           // 0 = B follows A, 1 = B is a random sentence
	};

	Parameters parameters;
	WordTokenizer tokenizer;
	std::optional<BertModel<T>> model;
	double lastTrainingLoss = 0.0;

	void requireTrained() const {
		if (!model) {
			throw PipelineStateError("ModelAdapter<BertModel>: model has not been trained.");
		}
	}

	static bool hasMaskedWord(const Example& example) {
		return std::any_of(example.mlmLabels.begin(), example.mlmLabels.end(),
			[](std::size_t label) { return label != 0; });
	}

	void validateMaskFraction() const {
		if (!(parameters.maskFraction > 0.0 && parameters.maskFraction <= 1.0)) {
			throw InvalidParameterError("BertParameters: maskFraction must be in (0, 1]!");
		}
	}

	/**
	 * @brief Turns sentences into word ids, cut to fit maximum length and
	 * leaving out ones without words.
	 */
	std::vector<std::vector<std::size_t>> encodeSentences(const std::vector<std::string>& sentences) const {
		// Room for [CLS] and two [SEP], shared by two sentences.
		const std::size_t maxWords = (parameters.maxLength - 3) / 2;

		std::vector<std::vector<std::size_t>> encoded;
		for (const std::string& sentence : sentences) {
			std::vector<std::size_t> ids = tokenizer.encode(sentence);
			if (ids.empty()) {
				continue;
			}
			if (ids.size() > maxWords) {
				ids.resize(maxWords);
			}
			encoded.push_back(std::move(ids));
		}
		return encoded;
	}

	/**
	 * @brief Builds one example from sentence `index`: pairs it with sentence
	 * that follows it or a random other one, and masks some of words.
	 */
	Example makeExample(const std::vector<std::vector<std::size_t>>& sentences, std::size_t index,
		std::mt19937& random) const {
		const std::size_t count = sentences.size();

		// Sentence B: next one half of time, otherwise a random other one.
		std::size_t second;
		if (index + 1 < count && std::bernoulli_distribution(0.5)(random)) {
			second = index + 1;
		}
		else {
			second = std::uniform_int_distribution<std::size_t>(0, count - 2)(random);
			if (second >= index) {
				second++; // skip `index` itself
			}
		}

		Example example;
		example.nspLabel = (second == index + 1) ? 0 : 1;

		const std::size_t cls = tokenizer.requireId("[CLS]");
		const std::size_t sep = tokenizer.requireId("[SEP]");
		const std::size_t mask = tokenizer.requireId("[MASK]");

		example.inputIds.push_back(cls);
		example.inputIds.insert(example.inputIds.end(), sentences[index].begin(), sentences[index].end());
		example.inputIds.push_back(sep);
		example.typeIds.assign(example.inputIds.size(), 0);
		example.inputIds.insert(example.inputIds.end(), sentences[second].begin(), sentences[second].end());
		example.inputIds.push_back(sep);
		example.typeIds.resize(example.inputIds.size(), 1);

		// Mask a fraction of words (never a special token), at least one.
		std::vector<std::size_t> candidates;
		for (std::size_t position = 0; position < example.inputIds.size(); position++) {
			if (!tokenizer.isSpecial(example.inputIds[position])) {
				candidates.push_back(position);
			}
		}
		std::shuffle(candidates.begin(), candidates.end(), random);
		const std::size_t maskCount = std::min(candidates.size(),
			std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(
				parameters.maskFraction * static_cast<double>(candidates.size())))));

		example.mlmLabels.assign(example.inputIds.size(), 0);
		for (std::size_t i = 0; i < maskCount; i++) {
			const std::size_t position = candidates[i];
			example.mlmLabels[position] = example.inputIds[position];
			example.inputIds[position] = mask;
		}
		return example;
	}
};
