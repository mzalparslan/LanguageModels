#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "CharTokenizer.h"
#include "Metrics.h"
#include "NGramBaseline.h"
#include "Validation.h"

/**
 * @brief How well a character-level language model predicts held-out text,
 * next to two baselines that need no neural network.
 *
 * A perplexity means little alone. Guessing uniformly scores vocabulary
 * size; letter frequencies (unigram) and one character of context (bigram)
 * score better, and a trained network that does not beat them is not learning
 * what it should.
 */
class LanguageModelMetrics {
public:
	// model's loss, perplexity, next-character accuracy and bits per character.
	Metrics model;
	// same for a unigram baseline counted from training text.
	Metrics unigram;
	// same for a bigram baseline counted from training text.
	Metrics bigram;
	// Number of characters scored.
	std::size_t characters = 0;
};

/**
 * @brief What character-level pipelines learn from training text: its
 * vocabulary, its ids, and baselines counted from it.
 */
class CharacterCorpus {
public:
	/**
	 * @brief Builds vocabulary and baselines from training text,
	 * replacing any earlier ones.
	 *
	 * @param trainingText Text to learn from.
	 * @param minimumLength Shortest text caller can train on.
	 * @throws InvalidSizeError If text is shorter than minimumLength.
	 */
	void fit(const std::vector<char>& trainingText, std::size_t minimumLength) {
		validation::requireAtLeast(trainingText.size(), minimumLength, "Training text length");

		tokenizer.fit(trainingText);
		ids = tokenizer.encode(trainingText);

		unigram.emplace(tokenizer.size(), 1, 1.0);
		bigram.emplace(tokenizer.size(), 2, 1.0);
		unigram->train(ids);
		bigram->train(ids);
	}

	bool isFitted() const { return unigram.has_value(); }

	std::size_t vocabSize() const { return tokenizer.size(); }

	const std::vector<std::size_t>& trainingIds() const { return ids; }

	/**
	 * @brief Converts text to ids with training vocabulary.
	 */
	std::vector<std::size_t> encode(const std::vector<char>& text) const { return tokenizer.encode(text); }

	/**
	 * @brief Scores baselines on held-out ids, filling every field of the
	 * result except `model`.
	 *
	 * @throws InvalidSizeError If fewer than two ids are given.
	 */
	LanguageModelMetrics scoreBaselines(const std::vector<std::size_t>& testIds) const {
		LanguageModelMetrics result;
		result.unigram = unigram->evaluate(testIds);
		result.bigram = bigram->evaluate(testIds);
		result.characters = testIds.size() - 1;
		return result;
	}

private:
	CharTokenizer tokenizer;
	std::vector<std::size_t> ids;
	std::optional<NGramBaseline> unigram;
	std::optional<NGramBaseline> bigram;
};
