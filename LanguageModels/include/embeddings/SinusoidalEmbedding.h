#pragma once

#include "Tensor.h"
#include "Validation.h"

/**
 * @brief Fixed (non-learned) sinusoidal positional encoding from "Attention
 * Is All You Need" (Vaswani et al., 2017). Adds a position-dependent vector
 * to every token embedding so model can tell positions apart.
 *
 * Nothing here is trainable, so gradient-related methods are no-ops and
 * exist only so class can stand in for a learned embedding.
 */
template <typename T>
class SinusoidalEmbedding {
public:
	/**
	 * @brief Read-only cache to hold positional encodings by sin/cos methods
	 * pe is calculated once in constructor and used as-is calculated.
	 */
	Tensor<T> pe;
	/**
	 * @brief Maximum sequence length model can handle.
	 * Model can handle sentences upto maxLen tokens long.
	 */
	std::size_t maxLen;
	/**
	 * @brief Dimension size of vector for each word.
	 * It must match d_model (embedding size) of Word Embeddings.
	 */
	std::size_t dModel;

	/**
	 * @brief Precomputes encoding table once.
	 *
	 * @param maxLength Longest sequence that can be encoded.
	 * @param dim Vector width; must match word embeddings' width.
	 */
	SinusoidalEmbedding(std::size_t maxLength, std::size_t dim)
		: maxLen(maxLength), dModel(dim) {
		validation::requirePositiveSize(maxLength, "SinusoidalEmbedding maximum length");
		validation::requirePositiveSize(dim, "SinusoidalEmbedding width");
		// Precompute PE matrix [maxLen, dModel]
		// Formula:
		//  PE(pos, 2i) = sin(pos / 10000^(2i/d))
		//  PE(pos, 2i + 1) = cos(pos / 10000^(2i/d))
		// Concept:
		// Provides a unique, continuous vector for every position that model can learn to use.
		pe = Tensor<T>({ maxLen, dModel }, T(0));

		for (std::size_t pos = 0; pos < maxLen; pos++) {
			// Even columns get sin, odd columns get cos of same frequency;
			// frequency falls geometrically from 1 to 1/10000 across dims,
			// so low dims encode fine position and high dims coarse position.
			for (std::size_t i = 0; i < dModel; i += 2) {
				double divTerm = std::exp(i * -std::log(10000.0) / dModel);

				pe.data[pos * dModel + i] = std::sin(pos * divTerm);
				if ((i + 1) < dModel) {
					pe.data[pos * dModel + i + 1] = std::cos(pos * divTerm);
				}
			}
		}
	}

	/**
	 * @brief Adds positional encoding to x in place.
	 *
	 * @param x Token embeddings [Seq, dModel]; row i receives pe row i.
	 * @throws InvalidSizeError If x is not a [Seq, dModel] matrix or Seq exceeds maxLen.
	 */
	void forward(Tensor<T>& x) {
		validation::requireColumns(x, dModel, "SinusoidalEmbedding input");
		std::size_t seq = x.shape[0];
		// should be d_model.
		std::size_t dim = x.shape[1];

		// Matches RotaryEmbedding::apply()'s policy: 
		// fail loudly instead of silently applying positional 
		// encoding to only part of sequence.
		if (seq > maxLen) {
			throw InvalidSizeError(
				"Input sequence length exceeds maximum "
				"length for Sinusoidal Embedding.");
		}

		// Row i of x is at position i, so add row i of table.
		for (std::size_t i = 0; i < seq; i++) {
			// Word Vector and PE should have same size to perform vector +.
			for (std::size_t j = 0; j < dim; j++) {
				x.data[i * dim + j] += pe.data[i * dim + j];
			}
		}
	}

	/**
	 * @brief No op.
	 */
	void zeroGrad() {}

	/**
	 * @brief No op.
	 */
	void backward(const std::vector<std::size_t>& x, 
		const Tensor<T>& dOut) {}

	/**
	 * @brief No op.
	 */
	void update(T lr) {}
};