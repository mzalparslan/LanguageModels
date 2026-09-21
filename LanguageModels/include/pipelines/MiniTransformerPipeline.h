#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "CudaRuntime.h"
#include "CudaVocabularyHead.h"
#include "Exceptions.h"
#include "LogitMetrics.h"
#include "Metrics.h"
#include "MiniTransformer.h"
#include "Pipeline.h"
#include "SentencePair.h"
#include "TextMetrics.h"
#include "TextNormalizer.h"
#include "Validation.h"
#include "WordTokenizer.h"

/**
 * @brief Settings of a translation model and of its training.
 */
template <typename T>
class TranslationParameters {
public:
	// Adam learning rate.
	T learningRate = T(0.001);
	// Passes over training pairs (one optimizer step per pair, batch size 1).
	std::size_t epochs = 500;
	// Most words model writes for one sentence when translating.
	std::size_t maxTranslationWords = 10;
	// Report average loss every this many epochs; 0 never reports.
	std::size_t logEveryEpochs = 100;
	// Threads used by ExecutionStrategy::Parallel; 0 uses hardware thread count.
	std::size_t threadCount = 0;
	// Seed of model's initial weights.
	std::uint32_t randomSeed = 42;
};

/**
 * @brief How well a translation model does on held-out sentence pairs.
 */
class TranslationMetrics {
public:
	// Loss, perplexity, accuracy and bits per word when model is fed the
	// correct earlier words (teacher forcing), as in training.
	Metrics teacherForced;
	// Corpus BLEU-4 of model's own translations against references
	// (0 to 1); 0 when no sentence shares a 4-word run with its reference.
	double bleu = 0.0;
	// Word edits needed to turn translations into references, divided
	// by reference word count. 0 is perfect.
	double wordErrorRate = 0.0;
	// Share of sentences translated exactly right.
	double exactMatchRate = 0.0;
	// Number of test pairs scored.
	std::size_t sentenceCount = 0;
};

/**
 * @brief Pipeline adapter for translation with MiniTransformer.
 *
 * Builds one word vocabulary per language from training pairs, trains the
 * model with teacher forcing (batch size 1, Adam), and scores it by BLEU, word
 * error rate and teacher-forced perplexity. With ExecutionStrategy::Parallel
 * every step is MiniTransformer::trainStepMultipleThread(); with Cuda it is
 * MiniTransformer::trainStepCuda(); otherwise MiniTransformer::trainStep().
 * Sequential and Parallel give the same trained model bit for bit, Cuda to within
 * rounding.
 */
template <typename T>
class ModelAdapter<T, MiniTransformer<T>, TranslationParameters<T>> {
public:
	using Sample = SentencePair;
	using Parameters = TranslationParameters<T>;
	using Result = TranslationMetrics;

	explicit ModelAdapter(Parameters parameters)
		: parameters(std::move(parameters)),
		sourceTokenizer({ unknownToken }, unknownToken),
		targetTokenizer({ unknownToken, startToken, endToken }, unknownToken) {}

	/**
	 * @brief Builds vocabularies from pairs, then trains a new model.
	 *
	 * @throws InvalidSizeError If no pair is usable (both sides must be
	 * non-empty and no longer than MiniTransformerConfig::maxSeqLen).
	 */
	void train(const std::vector<SentencePair>& pairs, ExecutionStrategy strategy, Logger& logger) {
		// vocabularies come from training pairs only.
		std::vector<std::string> sources, targets;
		sources.reserve(pairs.size());
		targets.reserve(pairs.size());
		for (const SentencePair& pair : pairs) {
			sources.push_back(pair.source);
			targets.push_back(pair.target);
		}

		sourceTokenizer.fit(sources);
		targetTokenizer.fit(targets);
		startId = targetTokenizer.requireId(startToken);
		endId = targetTokenizer.requireId(endToken);

		std::vector<Example> examples;
		std::size_t skipped = 0;
		for (const SentencePair& pair : pairs) {
			std::optional<Example> example = makeExample(pair);
			if (example) {
				examples.push_back(std::move(*example));
			}
			else {
				skipped++;
			}
		}
		if (skipped > 0) {
			logger.warning() << skipped << " training pairs skipped (empty or longer than "
				<< MiniTransformerConfig::maxSeqLen << " words).";
		}
		validation::requireNonEmpty(examples.size(), "Usable training pairs");

		logger.info() << "Source vocabulary: " << sourceTokenizer.size()
			<< ", target vocabulary: " << targetTokenizer.size()
			<< ", usable pairs: " << examples.size();

		model.emplace(sourceTokenizer.size(), targetTokenizer.size(), 100, parameters.randomSeed);

		// With ExecutionStrategy::Cuda the vocabulary-sized parameters live on the GPU
		// for the whole run and are copied back at the end.
		std::optional<cuda::VocabularyHead<T>> gpuHead;
		if (strategy == ExecutionStrategy::Cuda) {
			if (!cuda::isAvailable()) {
				throw CudaError("ExecutionStrategy::Cuda needs a CUDA device, and none is available.");
			}
			logger.info() << "GPU: " << cuda::describeDevice();
			gpuHead.emplace(model->createCudaHead());
		}

		std::size_t adamStep = 1; // Adam timestep: 1-based, increased after every example
		for (std::size_t epoch = 0; epoch < parameters.epochs; epoch++) {
			T totalLoss = T(0);
			for (const Example& example : examples) {
				const UpdateRule rule = UpdateRule::adam(adamStep);
				if (strategy == ExecutionStrategy::Cuda) {
					totalLoss += model->trainStepCuda(*gpuHead, example.source, example.decoderInput,
						example.labels, parameters.learningRate, rule);
				}
				else if (strategy == ExecutionStrategy::Parallel) {
					totalLoss += model->trainStepMultipleThread(example.source, example.decoderInput,
						example.labels, parameters.learningRate, rule, parameters.threadCount);
				}
				else {
					totalLoss += model->trainStep(example.source, example.decoderInput,
						example.labels, parameters.learningRate, rule);
				}
				adamStep++;
			}

			lastTrainingLoss = static_cast<double>(totalLoss) / static_cast<double>(examples.size());
			if (parameters.logEveryEpochs != 0 && epoch % parameters.logEveryEpochs == 0) {
				logger.info() << "Epoch " << epoch << " average loss per sentence: " << lastTrainingLoss;
			}
		}

		// Bring the trained weights back, so the model translates and is evaluated on the CPU.
		if (gpuHead) {
			model->downloadFromCudaHead(*gpuHead);
		}
	}

	/**
	 * @brief Translates held-out pairs and scores them.
	 *
	 * Pairs that are empty or longer than MiniTransformerConfig::maxSeqLen
	 * words are skipped.
	 *
	 * @throws PipelineStateError If model has not been trained.
	 * @throws InvalidSizeError If no pair is usable.
	 */
	TranslationMetrics evaluate(const std::vector<SentencePair>& pairs, Logger& logger) {
		requireTrained();

		evaluation::MetricsAccumulator teacherForced;
		std::vector<evaluation::Tokens> candidates;
		std::vector<std::vector<evaluation::Tokens>> references;
		std::size_t editCount = 0;
		std::size_t referenceWordCount = 0;
		std::size_t exactMatches = 0;
		std::size_t skipped = 0;

		for (const SentencePair& pair : pairs) {
			std::optional<Example> example = makeExample(pair);
			if (!example) {
				skipped++;
				continue;
			}

			// Teacher forced: how likely are right words, given right earlier words.
			Tensor<T> logits;
			model->forward(example->source, example->decoderInput, logits);
			teacherForced.addLogits(logits, example->labels);

			// Free-running: model's own translation.
			evaluation::Tokens candidate = translateWords(example->source);
			evaluation::Tokens reference = text::splitOnWhitespace(pair.target);

			editCount += evaluation::editDistance(candidate, reference);
			referenceWordCount += reference.size();
			if (candidate == reference) {
				exactMatches++;
			}
			candidates.push_back(std::move(candidate));
			references.push_back({ std::move(reference) });
		}
		if (skipped > 0) {
			logger.warning() << skipped << " test pairs skipped (empty or longer than "
				<< MiniTransformerConfig::maxSeqLen << " words).";
		}
		validation::requireNonEmpty(candidates.size(), "Usable test pairs");

		TranslationMetrics result;
		result.teacherForced = teacherForced.result();
		result.bleu = evaluation::corpusBleu(candidates, references).bleu;
		result.wordErrorRate = static_cast<double>(editCount) / static_cast<double>(referenceWordCount);
		result.exactMatchRate = static_cast<double>(exactMatches) / static_cast<double>(candidates.size());
		result.sentenceCount = candidates.size();
		return result;
	}

	/**
	 * @brief Translates one sentence with greedy decoding.
	 *
	 * @param sentence Source sentence; normalized like training data.
	 * @return translation, words separated by spaces.
	 * @throws PipelineStateError If model has not been trained.
	 * @throws InvalidSizeError If sentence has no words, or too many.
	 */
	std::string translate(const std::string& sentence) {
		requireTrained();

		const std::vector<std::size_t> source = sourceTokenizer.encode(text::normalize(sentence));
		validation::requireNonEmpty(source.size(), "Sentence to translate");
		validation::requireAtMost(source.size(), MiniTransformerConfig::maxSeqLen, "Sentence length");

		std::string translation;
		for (const std::string& word : translateWords(source)) {
			translation += (translation.empty() ? "" : " ") + word;
		}
		return translation;
	}

	/**
	 * @brief trained model.
	 *
	 * @throws PipelineStateError If model has not been trained.
	 */
	MiniTransformer<T>& trainedModel() {
		requireTrained();
		return *model;
	}

	/**
	 * @brief Average loss per training pair over last epoch trained.
	 */
	double finalTrainingLoss() const { return lastTrainingLoss; }

private:
	// A training or test pair as model input.
	class Example {
	public:
		std::vector<std::size_t> source;
		std::vector<std::size_t> decoderInput; // <SOS> w1 .. wn
		std::vector<std::size_t> labels;       // w1 .. wn <EOS>
	};

	static inline const std::string unknownToken = "<UNK>";
	static inline const std::string startToken = "<SOS>";
	static inline const std::string endToken = "<EOS>";

	Parameters parameters;
	WordTokenizer sourceTokenizer;
	WordTokenizer targetTokenizer;
	std::optional<MiniTransformer<T>> model;
	std::size_t startId = 0;
	std::size_t endId = 0;
	double lastTrainingLoss = 0.0;

	void requireTrained() const {
		if (!model) {
			throw PipelineStateError("ModelAdapter<MiniTransformer>: model has not been trained.");
		}
	}

	/**
	 * @brief Turns a pair into model input, or nothing if it cannot be used.
	 */
	std::optional<Example> makeExample(const SentencePair& pair) const {
		Example example;
		example.source = sourceTokenizer.encode(pair.source);
		const std::vector<std::size_t> words = targetTokenizer.encode(pair.target);
		if (example.source.empty() || words.empty()) {
			return std::nullopt;
		}

		// Teacher forcing: decoder is fed <SOS> and target words, and must
		// predict at each position next word, ending with <EOS>.
		example.decoderInput.push_back(startId);
		example.decoderInput.insert(example.decoderInput.end(), words.begin(), words.end());
		example.labels = words;
		example.labels.push_back(endId);

		if (example.source.size() > MiniTransformerConfig::maxSeqLen
			|| example.decoderInput.size() > MiniTransformerConfig::maxSeqLen) {
			return std::nullopt;
		}
		return example;
	}

	/**
	 * @brief Greedy translation as words. start and end tokens are left
	 * out, but an <UNK> model writes stays, so it counts as mistake it is.
	 */
	evaluation::Tokens translateWords(const std::vector<std::size_t>& source) {
		evaluation::Tokens words;
		for (std::size_t tokenId : model->generate(source, startId, endId, parameters.maxTranslationWords)) {
			if (tokenId != startId && tokenId != endId) {
				words.push_back(targetTokenizer.word(tokenId));
			}
		}
		return words;
	}
};
