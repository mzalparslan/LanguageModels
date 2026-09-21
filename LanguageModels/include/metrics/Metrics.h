#pragma once

#include <cmath>
#include <cstddef>
#include <string>

#include "Exceptions.h"

/**
 * @brief Evaluation results for a model that predicts next token.
 *
 * Independent of any particular model: it only needs totals accumulated
 * while scoring a sequence (summed cross-entropy, number of correct
 * predictions, number of predictions), so any language model can report its
 * results with it. Build one with fromTotals().
 */
class Metrics {
public:
	/**
	 * @brief Loss ratio: mean cross-entropy per prediction, in nats.
	 */
	double loss;
	/**
	 * @brief PPL: Measures how well probability distribution predicts a sample.
	 * Closer to 1.0 is better.
	 */
	double perplexity;
	/**
	 * @brief Acc: fraction (0..1) of times model correctly predicted exact next token.
	 */
	double accuracy;
	/**
	 * @brief BPC (Bits Per Character): Loss in base-2, useful for compression comparison.
	 * Standart metric used to measure a model's ability to compress data.
	 * It connects Language Modeling to Information Theory.
	 * 1. Comparison with Compression Algorithms:
	 * * Standard ASCII text uses 8-bits per character.
	 * * A standard zip file might get text down to ~2-3 bits per character.
	 * * State-of-the-art Language Models (like GPT's base models) can get English text
	 * down to ~0.9 ~1.1 bits per character.
	 * By calculating BPC, you can see if your NN is smarter than a standard compression algorithm.
	 * 2. Model Independence
	 * Perplexity depends on base (2 or 10) and can be hard to intuit (is 45 good? or 120 bad?)
	 * BPC is always in "bits" which is universal unit. It allows you to compare
	 * a character-level RNN against a word-level Transformer (by converting word-bits to character-bits)
	 * or even against a ZIP file.
	 * RESULT: Your model has effectively memorized string "hello world".
	 * it needs almost 0 bits of information to guess next char because it is 100% certain.
	 * It has achieved near-perfect compression for that specific string.
	 */
	double bpc;

	/**
	 * @brief Derives all four metrics from raw totals.
	 *
	 * @param totalLoss Sum of -ln(p(target)) over every prediction.
	 * @param correctPredictions Number of predictions whose arg-max was target.
	 * @param predictionCount Number of predictions made. Must be non-zero.
	 * @throws DivisionByZeroError If predictionCount is 0.
	 * @throws InvalidParameterError If correctPredictions exceeds predictionCount.
	 * @throws NaNError, NonFiniteError If totalLoss is not finite.
	 */
	static Metrics fromTotals(double totalLoss, std::size_t correctPredictions,
		std::size_t predictionCount) {
		if (predictionCount == 0) {
			throw DivisionByZeroError("Division by zero: prediction count is zero!");
		}
		if (correctPredictions > predictionCount) {
			throw InvalidParameterError("Correct predictions exceed prediction count!");
		}
		if (std::isnan(totalLoss)) {
			throw NaNError("Total loss is NaN!");
		}
		if (!std::isfinite(totalLoss)) {
			throw NonFiniteError("Total loss is infinite!");
		}

		Metrics m;
		m.loss = totalLoss / (double)predictionCount;
		m.perplexity = std::exp(m.loss);
		m.accuracy = (double)correctPredictions / (double)predictionCount;
		m.bpc = m.loss / std::log(2.0);

		return m;
	}
};
