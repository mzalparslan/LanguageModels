#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "Metrics.h"
#include "Tensor.h"
#include "Validation.h"

/**
 * @brief Model-agnostic scoring of next-token predictions, straight from a
 * model's output logits.
 *
 * Every model in this library that predicts tokens (DecoderOnlyModel,
 * MiniTransformer, BERT's masked-language head, ...) produces a
 * [rows, vocab] logits tensor, so these functions let any of them be
 * evaluated same way and compared on equal terms.
 */
namespace evaluation {

	/**
	 * @brief Sums cross-entropy and hit counts over any number of scored
	 * batches (windows of a long text, validation examples, ...) and turns the
	 * totals into one Metrics at end.
	 *
	 * Totals are accumulated, not averages of averages, so batches of
	 * different sizes are weighted correctly.
	 */
	class MetricsAccumulator {
	public:
		/**
		 * @brief Adds raw totals, e.g. from a model that computes its own loss.
		 *
		 * @param totalLoss Sum of -ln(p(target)) over predictions.
		 * @param correctPredictions Number of predictions whose arg-max was target.
		 * @param predictionCount Number of predictions made.
		 * @throws InvalidParameterError If correctPredictions exceeds predictionCount.
		 */
		void add(double totalLoss, std::size_t correctPredictions, std::size_t predictionCount) {
			if (correctPredictions > predictionCount) {
				throw InvalidParameterError("Correct predictions exceed prediction count!");
			}
			lossSum += totalLoss;
			correctCount += correctPredictions;
			predictionTotal += predictionCount;
		}

		/**
		 * @brief Scores every row of a logits matrix against its target token.
		 *
		 * Cross-entropy uses a numerically stable log-softmax, so large
		 * logits do not overflow. A tie for highest logit counts as
		 * a hit only when target is lowest-numbered tied token.
		 *
		 * @param logits Model output [rows, vocab]; row i scores token at target i.
		 * @param targets Expected token id for each row.
		 * @throws InvalidSizeError If logits is not a non-empty matrix or targets
		 * differs in length from its row count.
		 * @throws InvalidParameterError If a target id is outside vocabulary.
		 */
		template <typename T>
		void addLogits(const Tensor<T>& logits, const std::vector<std::size_t>& targets) {
			validation::requireMatrix(logits, "Logits");
			validation::requireSameSize(targets.size(), logits.shape[0], "Target sequence");

			const std::size_t rows = logits.shape[0];
			const std::size_t vocab = logits.shape[1];

			double batchLoss = 0.0;
			std::size_t batchCorrect = 0;

			for (std::size_t row = 0; row < rows; row++) {
				const std::size_t target = targets[row];
				validation::requireBelow(target, vocab, "Target token id");

				const T* scores = &logits.data[row * vocab];

				std::size_t best = 0;
				for (std::size_t j = 1; j < vocab; j++) {
					if (scores[j] > scores[best]) {
						best = j;
					}
				}

				// -log softmax(target) = log(sum exp(x - max)) + max - x[target]
				const double maxScore = static_cast<double>(scores[best]);
				double sum = 0.0;
				for (std::size_t j = 0; j < vocab; j++) {
					sum += std::exp(static_cast<double>(scores[j]) - maxScore);
				}
				batchLoss += std::log(sum) + maxScore - static_cast<double>(scores[target]);

				if (best == target) {
					batchCorrect++;
				}
			}

			add(batchLoss, batchCorrect, rows);
		}

		/**
		 * @brief Number of predictions added so far.
		 */
		std::size_t count() const { return predictionTotal; }

		/**
		 * @brief Loss, perplexity, accuracy and bits per token over everything added.
		 *
		 * @throws DivisionByZeroError If nothing was added.
		 * @throws NaNError, NonFiniteError If accumulated loss is not finite.
		 */
		Metrics result() const {
			return Metrics::fromTotals(lossSum, correctCount, predictionTotal);
		}

	private:
		double lossSum = 0.0;
		std::size_t correctCount = 0;
		std::size_t predictionTotal = 0;
	};

	/**
	 * @brief Loss, perplexity, accuracy and bits per token of one logits matrix.
	 *
	 * @see MetricsAccumulator::addLogits for details and exceptions.
	 */
	template <typename T>
	Metrics scoreLogits(const Tensor<T>& logits, const std::vector<std::size_t>& targets) {
		MetricsAccumulator accumulator;
		accumulator.addLogits(logits, targets);
		return accumulator.result();
	}

	/**
	 * @brief Fraction of rows whose target is among k highest-scoring tokens.
	 *
	 * k = 1 is ordinary accuracy. Ties are resolved in model's favour: a
	 * target counts as a hit when fewer than k tokens score strictly higher.
	 *
	 * @param logits Model output [rows, vocab].
	 * @param targets Expected token id for each row.
	 * @param k How many top-scoring tokens count as a hit; 1 <= k <= vocab.
	 * @throws InvalidSizeError If logits is not a non-empty matrix, targets differs
	 * in length from its row count, or k exceeds vocabulary size.
	 * @throws InvalidParameterSizeError If k is zero.
	 * @throws InvalidParameterError If a target id is outside vocabulary.
	 */
	template <typename T>
	double topKAccuracy(const Tensor<T>& logits, const std::vector<std::size_t>& targets,
		std::size_t k) {
		validation::requireMatrix(logits, "Logits");
		validation::requireSameSize(targets.size(), logits.shape[0], "Target sequence");

		const std::size_t rows = logits.shape[0];
		const std::size_t vocab = logits.shape[1];
		validation::requirePositiveSize(k, "Top-k size");
		validation::requireAtMost(k, vocab, "Top-k size");

		std::size_t hits = 0;
		for (std::size_t row = 0; row < rows; row++) {
			const std::size_t target = targets[row];
			validation::requireBelow(target, vocab, "Target token id");

			const T* scores = &logits.data[row * vocab];
			std::size_t higher = 0;
			for (std::size_t j = 0; j < vocab; j++) {
				if (scores[j] > scores[target]) {
					higher++;
				}
			}
			if (higher < k) {
				hits++;
			}
		}

		return static_cast<double>(hits) / static_cast<double>(rows);
	}

} // namespace evaluation
