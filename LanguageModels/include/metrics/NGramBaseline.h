#pragma once

#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "Metrics.h"
#include "Validation.h"

/**
 * @brief Count-based unigram or bigram language model, used as a yardstick.
 *
 * A neural model's perplexity means little on its own; it becomes meaningful
 * next to models whose score can be worked out independently:
 *
 *  - guessing uniformly over vocabulary gives perplexity == vocabulary size;
 *  - a unigram model (letter frequencies) is entropy of text;
 *  - a bigram model (one character of context) is conditional entropy.
 *
 * A trained network that does not beat these is not learning what it should.
 * Counts are stored sparsely, so a word-level vocabulary is fine.
 */
class NGramBaseline {
public:
	/**
	 * @param vocabSize Number of token ids; ids are 0 .. vocabSize - 1.
	 * @param order 1 for a unigram model (no context), 2 for a bigram model
	 * (the previous token is context).
	 * @param smoothing Add-k smoothing: k is added to every count, so unseen
	 * tokens keep a small probability. Use 0 for plain maximum-likelihood
	 * estimate (which gives infinite loss on a token never seen in training),
	 * or a positive value, e.g. 1 (Laplace), to score held-out text.
	 * @throws InvalidParameterSizeError If vocabSize is zero.
	 * @throws InvalidParameterError If order is not 1 or 2, or smoothing is negative.
	 */
	NGramBaseline(std::size_t vocabSize, std::size_t order, double smoothing = 1.0)
		: vocab(vocabSize), order(order), smoothing(smoothing),
		unigramCounts(vocabSize, 0), followers(vocabSize), contextTotals(vocabSize, 0)
	{
		validation::requirePositiveSize(vocabSize, "Baseline vocabulary size");
		if (order != 1 && order != 2) {
			throw InvalidParameterError("Baseline order must be 1 (unigram) or 2 (bigram)!");
		}
		validation::requireNonNegativeFinite(smoothing, "Baseline smoothing");
	}

	/**
	 * @brief Adds a token sequence to counts. May be called repeatedly.
	 *
	 * @throws InvalidParameterError If a token id is outside vocabulary.
	 */
	void train(const std::vector<std::size_t>& tokens) {
		for (std::size_t token : tokens) {
			validation::requireBelow(token, vocab, "Token id");
		}
		for (std::size_t i = 0; i < tokens.size(); i++) {
			unigramCounts[tokens[i]]++;
			totalTokens++;
			if (i + 1 < tokens.size()) {
				followers[tokens[i]][tokens[i + 1]]++;
				contextTotals[tokens[i]]++;
			}
		}
	}

	/**
	 * @brief Probability of `next` given previous token (ignored by a
	 * unigram model).
	 *
	 * @throws InvalidParameterError If an id is outside vocabulary.
	 * @throws DivisionByZeroError If nothing was trained and smoothing is 0.
	 */
	double probability(std::size_t previous, std::size_t next) const {
		validation::requireBelow(previous, vocab, "Context token id");
		validation::requireBelow(next, vocab, "Token id");

		double count;
		double total;
		if (order == 1) {
			count = static_cast<double>(unigramCounts[next]);
			total = static_cast<double>(totalTokens);
		}
		else {
			auto found = followers[previous].find(next);
			count = found == followers[previous].end() ? 0.0 : static_cast<double>(found->second);
			total = static_cast<double>(contextTotals[previous]);
		}

		const double denominator = total + smoothing * static_cast<double>(vocab);
		validation::requireNonZeroDenominator(denominator, "Baseline probability denominator");
		return (count + smoothing) / denominator;
	}

	/**
	 * @brief Scores a sequence way a next-token model is scored: each
	 * token predicts one after it, so a sequence of n tokens makes n - 1
	 * predictions. prediction is most frequent token (lowest id on a tie).
	 *
	 * @return Loss, perplexity, accuracy and bits per token.
	 * @throws InvalidSizeError If fewer than two tokens are given.
	 * @throws InvalidParameterError If a token id is outside vocabulary.
	 * @throws NonFiniteError If a token has probability 0 (see smoothing).
	 */
	Metrics evaluate(const std::vector<std::size_t>& tokens) const {
		if (tokens.size() < 2) {
			throw InvalidSizeError("Baseline evaluation needs at least two tokens!");
		}
		for (std::size_t token : tokens) {
			validation::requireBelow(token, vocab, "Token id");
		}

		std::vector<std::size_t> bestNext = mostFrequentNext();

		double totalLoss = 0.0;
		std::size_t correct = 0;
		for (std::size_t i = 0; i + 1 < tokens.size(); i++) {
			totalLoss -= std::log(probability(tokens[i], tokens[i + 1]));
			const std::size_t predicted = order == 1 ? bestNext[0] : bestNext[tokens[i]];
			if (predicted == tokens[i + 1]) {
				correct++;
			}
		}

		return Metrics::fromTotals(totalLoss, correct, tokens.size() - 1);
	}

private:
	std::size_t vocab;
	std::size_t order;
	double smoothing;
	std::size_t totalTokens = 0;
	std::vector<std::size_t> unigramCounts;
	// followers[a][b] = how often token b came directly after token a.
	std::vector<std::unordered_map<std::size_t, std::size_t>> followers;
	std::vector<std::size_t> contextTotals;

	/**
	 * @brief Arg-max prediction per context (index 0 only, for a unigram model).
	 * Contexts never seen in training predict token 0, as all counts tie there.
	 */
	std::vector<std::size_t> mostFrequentNext() const {
		std::vector<std::size_t> best(vocab, 0);
		if (order == 1) {
			for (std::size_t token = 1; token < vocab; token++) {
				if (unigramCounts[token] > unigramCounts[best[0]]) {
					best[0] = token;
				}
			}
			return best;
		}

		for (std::size_t context = 0; context < vocab; context++) {
			std::size_t bestCount = 0;
			for (const auto& [next, count] : followers[context]) {
				if (count > bestCount || (count == bestCount && next < best[context])) {
					best[context] = next;
					bestCount = count;
				}
			}
		}
		return best;
	}
};
