#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "Exceptions.h"
#include "Validation.h"

/**
 * @brief Reference-based scores for generated text: edit distance (word and
 * character error rate), BLEU and ROUGE.
 *
 * These compare a model's output with one or more human references, which is
 * how translation (the Mini Transformer's French-to-English task) and
 * summarisation models are judged, since next-token perplexity does not say
 * whether a whole generated sentence is right. All of them work on token
 * sequences, so they do not depend on any tokenizer.
 */
namespace evaluation {

	using Tokens = std::vector<std::string>;

	/**
	 * @brief Lower-cases text and splits it into words on whitespace,
	 * trimming punctuation from both ends of each word.
	 *
	 * A deliberately simple tokenizer for scoring: "Party." and "party"
	 * become same word. Use your own tokenizer when task needs one.
	 */
	inline Tokens splitWords(const std::string& text) {
		Tokens words;
		std::string word;

		auto flush = [&]() {
			std::size_t begin = 0;
			std::size_t end = word.size();
			while (begin < end && std::ispunct(static_cast<unsigned char>(word[begin]))) {
				begin++;
			}
			while (end > begin && std::ispunct(static_cast<unsigned char>(word[end - 1]))) {
				end--;
			}
			if (end > begin) {
				words.push_back(word.substr(begin, end - begin));
			}
			word.clear();
		};

		for (char c : text) {
			if (std::isspace(static_cast<unsigned char>(c))) {
				flush();
			}
			else {
				word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
			}
		}
		flush();

		return words;
	}

	/**
	 * @brief Levenshtein distance: fewest insertions, deletions and
	 * substitutions that turn one sequence into other.
	 */
	template <typename T>
	std::size_t editDistance(const std::vector<T>& a, const std::vector<T>& b) {
		// One row of classic dynamic-programming table at a time.
		std::vector<std::size_t> previous(b.size() + 1);
		std::vector<std::size_t> current(b.size() + 1);
		for (std::size_t j = 0; j <= b.size(); j++) {
			previous[j] = j;
		}

		for (std::size_t i = 1; i <= a.size(); i++) {
			current[0] = i;
			for (std::size_t j = 1; j <= b.size(); j++) {
				const std::size_t substitution = previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
				current[j] = std::min({ substitution, previous[j] + 1, current[j - 1] + 1 });
			}
			std::swap(previous, current);
		}

		return previous[b.size()];
	}

	/**
	 * @brief Levenshtein distance between two strings, character by character.
	 */
	inline std::size_t editDistance(const std::string& a, const std::string& b) {
		return editDistance(std::vector<char>(a.begin(), a.end()), std::vector<char>(b.begin(), b.end()));
	}

	/**
	 * @brief Word error rate: word-level edit distance divided by the
	 * reference length. 0 is a perfect match; it can exceed 1 when the
	 * hypothesis is much longer than reference.
	 *
	 * @throws DivisionByZeroError If reference is empty.
	 */
	inline double wordErrorRate(const Tokens& reference, const Tokens& hypothesis) {
		validation::requireNonZeroDenominator(reference.size(), "Reference word count");
		return static_cast<double>(editDistance(hypothesis, reference))
			/ static_cast<double>(reference.size());
	}

	/**
	 * @brief Character error rate: character-level edit distance divided by
	 * reference length.
	 *
	 * @throws DivisionByZeroError If reference is empty.
	 */
	inline double characterErrorRate(const std::string& reference, const std::string& hypothesis) {
		validation::requireNonZeroDenominator(reference.size(), "Reference character count");
		return static_cast<double>(editDistance(hypothesis, reference))
			/ static_cast<double>(reference.size());
	}

	/**
	 * @brief BLEU (Papineni et al., 2002) with its components.
	 */
	class BleuScore {
	public:
		/**
		 * @brief score, from 0 to 1. 0 when any n-gram precision is 0
		 * (unless smoothing was requested).
		 */
		double bleu = 0.0;
		/**
		 * @brief Modified (clipped) n-gram precision for n = 1 .. maxN.
		 */
		std::vector<double> precisions;
		/**
		 * @brief Penalty for candidates shorter than their references (1 = none).
		 */
		double brevityPenalty = 0.0;
		/**
		 * @brief Total candidate length in tokens.
		 */
		std::size_t candidateLength = 0;
		/**
		 * @brief Total length of closest-length reference of each candidate.
		 */
		std::size_t referenceLength = 0;
	};

	namespace detail {

		using NGramCounts = std::map<Tokens, std::size_t>;

		inline NGramCounts countNGrams(const Tokens& tokens, std::size_t n) {
			NGramCounts counts;
			if (tokens.size() >= n) {
				for (std::size_t i = 0; i + n <= tokens.size(); i++) {
					counts[Tokens(tokens.begin() + i, tokens.begin() + i + n)]++;
				}
			}
			return counts;
		}

		inline std::size_t clippedMatches(const NGramCounts& candidate, const NGramCounts& reference) {
			std::size_t matches = 0;
			for (const auto& [ngram, count] : candidate) {
				auto found = reference.find(ngram);
				if (found != reference.end()) {
					matches += std::min(count, found->second);
				}
			}
			return matches;
		}

		inline std::size_t longestCommonSubsequence(const Tokens& a, const Tokens& b) {
			std::vector<std::size_t> previous(b.size() + 1, 0);
			std::vector<std::size_t> current(b.size() + 1, 0);
			for (std::size_t i = 1; i <= a.size(); i++) {
				for (std::size_t j = 1; j <= b.size(); j++) {
					current[j] = a[i - 1] == b[j - 1]
						? previous[j - 1] + 1
						: std::max(previous[j], current[j - 1]);
				}
				std::swap(previous, current);
			}
			return previous[b.size()];
		}

	} // namespace detail

	/**
	 * @brief Corpus-level BLEU: n-gram matches and lengths are summed over the
	 * whole corpus before ratios are taken, which is how BLEU is
	 * normally reported (averaging per-sentence scores is not same).
	 *
	 * Precision counts are clipped: an n-gram is credited at most as often as
	 * it appears in a single reference. brevity penalty uses the
	 * reference length closest to each candidate (the shorter one on a tie).
	 *
	 * @param candidates One generated sentence per example.
	 * @param references For each candidate, one or more reference sentences.
	 * @param maxN Highest n-gram order; 4 is standard (BLEU-4).
	 * @param smooth Add 1 to matches and totals of orders above 1, so a
	 * sentence without a matching 4-gram scores above 0 instead of exactly 0
	 * (Lin and Och's smoothing).
	 * @throws InvalidSizeError If there are no candidates, or references differs in
	 * length from candidates, or a candidate has no reference.
	 * @throws InvalidParameterSizeError If maxN is zero.
	 */
	inline BleuScore corpusBleu(const std::vector<Tokens>& candidates,
		const std::vector<std::vector<Tokens>>& references,
		std::size_t maxN = 4, bool smooth = false) {
		validation::requireNonEmpty(candidates.size(), "BLEU candidate list");
		validation::requireSameSize(references.size(), candidates.size(), "BLEU reference list");
		validation::requirePositiveSize(maxN, "BLEU maximum n-gram order");

		std::vector<std::size_t> matches(maxN, 0);
		std::vector<std::size_t> totals(maxN, 0);
		BleuScore score;

		for (std::size_t i = 0; i < candidates.size(); i++) {
			validation::requireNonEmpty(references[i].size(), "BLEU reference set");
			const Tokens& candidate = candidates[i];

			for (std::size_t n = 1; n <= maxN; n++) {
				const detail::NGramCounts candidateCounts = detail::countNGrams(candidate, n);

				// Each n-gram may be credited up to its highest count in any one reference.
				detail::NGramCounts clip;
				for (const Tokens& reference : references[i]) {
					for (const auto& [ngram, count] : detail::countNGrams(reference, n)) {
						std::size_t& best = clip[ngram];
						best = std::max(best, count);
					}
				}

				matches[n - 1] += detail::clippedMatches(candidateCounts, clip);
				totals[n - 1] += candidate.size() >= n ? candidate.size() - n + 1 : 0;
			}

			// Reference whose length is closest to candidate's (shorter on a tie).
			std::size_t closest = references[i][0].size();
			for (const Tokens& reference : references[i]) {
				const std::size_t distance = reference.size() > candidate.size()
					? reference.size() - candidate.size() : candidate.size() - reference.size();
				const std::size_t bestDistance = closest > candidate.size()
					? closest - candidate.size() : candidate.size() - closest;
				if (distance < bestDistance || (distance == bestDistance && reference.size() < closest)) {
					closest = reference.size();
				}
			}
			score.candidateLength += candidate.size();
			score.referenceLength += closest;
		}

		double logPrecisionSum = 0.0;
		bool anyZero = false;
		for (std::size_t n = 1; n <= maxN; n++) {
			double numerator = static_cast<double>(matches[n - 1]);
			double denominator = static_cast<double>(totals[n - 1]);
			if (smooth && n > 1) {
				numerator += 1.0;
				denominator += 1.0;
			}
			const double precision = denominator == 0.0 ? 0.0 : numerator / denominator;
			score.precisions.push_back(precision);
			if (precision == 0.0) {
				anyZero = true;
			}
			else {
				logPrecisionSum += std::log(precision);
			}
		}

		if (score.candidateLength == 0) {
			score.brevityPenalty = 0.0;
		}
		else if (score.candidateLength > score.referenceLength) {
			score.brevityPenalty = 1.0;
		}
		else {
			score.brevityPenalty = std::exp(1.0 - static_cast<double>(score.referenceLength)
				/ static_cast<double>(score.candidateLength));
		}

		score.bleu = anyZero ? 0.0
			: score.brevityPenalty * std::exp(logPrecisionSum / static_cast<double>(maxN));
		return score;
	}

	/**
	 * @brief BLEU of a single sentence against one or more references.
	 *
	 * Sentence-level BLEU is 0 whenever a sentence lacks a matching n-gram of
	 * some order, so short sentences usually want smooth = true.
	 *
	 * @see corpusBleu
	 */
	inline BleuScore sentenceBleu(const Tokens& candidate, const std::vector<Tokens>& references,
		std::size_t maxN = 4, bool smooth = false) {
		return corpusBleu({ candidate }, { references }, maxN, smooth);
	}

	/**
	 * @brief Precision, recall and F1 of a ROUGE score.
	 */
	class OverlapScore {
	public:
		/**
		 * @brief Share of candidate that appears in reference.
		 */
		double precision = 0.0;
		/**
		 * @brief Share of reference that appears in candidate.
		 */
		double recall = 0.0;
		/**
		 * @brief Harmonic mean of precision and recall.
		 */
		double f1 = 0.0;

		/**
		 * @brief Builds a score from an overlap size and two lengths.
		 * Every ratio is 0 when its denominator is 0.
		 */
		static OverlapScore fromCounts(std::size_t overlap, std::size_t candidateSize,
			std::size_t referenceSize) {
			OverlapScore score;
			score.precision = candidateSize == 0 ? 0.0
				: static_cast<double>(overlap) / static_cast<double>(candidateSize);
			score.recall = referenceSize == 0 ? 0.0
				: static_cast<double>(overlap) / static_cast<double>(referenceSize);
			score.f1 = (score.precision + score.recall) == 0.0 ? 0.0
				: 2.0 * score.precision * score.recall / (score.precision + score.recall);
			return score;
		}
	};

	/**
	 * @brief ROUGE-N (Lin, 2004): overlap of n-grams between a candidate and a
	 * reference. ROUGE-1 counts words, ROUGE-2 word pairs.
	 *
	 * @throws InvalidParameterSizeError If n is zero.
	 */
	inline OverlapScore rougeN(const Tokens& candidate, const Tokens& reference, std::size_t n) {
		validation::requirePositiveSize(n, "ROUGE n-gram order");

		const detail::NGramCounts candidateCounts = detail::countNGrams(candidate, n);
		const detail::NGramCounts referenceCounts = detail::countNGrams(reference, n);

		return OverlapScore::fromCounts(
			detail::clippedMatches(candidateCounts, referenceCounts),
			candidate.size() >= n ? candidate.size() - n + 1 : 0,
			reference.size() >= n ? reference.size() - n + 1 : 0);
	}

	/**
	 * @brief ROUGE-L (Lin, 2004): based on longest common subsequence,
	 * so it rewards words that appear in same order without needing
	 * them to be adjacent.
	 */
	inline OverlapScore rougeL(const Tokens& candidate, const Tokens& reference) {
		return OverlapScore::fromCounts(
			detail::longestCommonSubsequence(candidate, reference),
			candidate.size(), reference.size());
	}

} // namespace evaluation
