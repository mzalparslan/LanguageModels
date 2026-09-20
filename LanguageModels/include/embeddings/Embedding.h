#pragma once

#include "Parameter.h"

/**
 * @brief Learned token embedding.
 *
 * Embedded Matrix maps a Token (Integer) to a Vector (list of numbers).
 * It is a dictionary where definition is a set of coordinates in math space.
 * vectors are trained along with rest of model.
 */
template <typename T>
class Embedding {
public:
	/**
	 * @brief Embedding Matrix (vocabSize, dModel). Row[i] stores vector
	 * for word ID i.
	 */
	Parameter<T> table;
	/**
	 * @brief Total number of unique words (tokens) model knows.
	 */
	std::size_t vocabSize;
	/**
	 * @brief Dimension size of vector for each word.
	 * Determines number of Columns in lookup table.
	 * "Richness" of meaning.
	 * If dModel = 32 means every word is represented by 32 floating point numbers.
	 * e.g., GPT-3 uses ~12288.
	 */
	std::size_t dModel;

	/**
	 * @brief Creates lookup table with small random vectors.
	 *
	 * @param vocab Number of distinct token ids (rows).
	 * @param dim Vector width per token (columns).
	 * @param rng Engine initial vectors are drawn from.
	 */
	Embedding(std::size_t vocab, std::size_t dim,
		RandomEngine& rng)
		: vocabSize(vocab), dModel(dim) {
		validation::requirePositiveSize(vocab, "Embedding vocabulary size");
		validation::requirePositiveSize(dim, "Embedding width");
		table.init({ vocabSize, dModel }, T(0.1), rng);
	}

	/**
	 * @brief Zero all gradients current embedding table.
	 *
	 * Should be called at early start of every training step before
	 * and backward or Forward passes for that new batch of data.
	 */
	void zeroGrad() {
		table.zeroGrad();
	}

	/**
	 * @brief Lookup Table:
	 * exm: if vocabSize = 4 and dModel = 3:
	 * Row Index	Word		Vector
	 *    0			"hello"		[0.1, -0.5, 0.9]
	 *    1         "world"     [0.8, 0.2, -0.1]
	 *    2         "cat"       [-0.4, 0.3, 0.1]
	 *    3         "dog"       [-0.3, 0.4, 0.2]
	 * Input token id 2 = "cat"
	 * Model looks up Row #2: [-0.4, 0.3, 0.1]
	 * Transformer will do math on these vectors and see that
	 * "cat" and "dog" are close to each other whereas
	 * "hello" is far away. If just used integers 2 and 3, math would fail.
	 */
	void forward(const std::vector<std::size_t>& x, Tensor<T>& out) {
		validation::requireNonEmpty(x.size(), "Token id sequence");
		std::size_t seq = x.size();

		out = Tensor<T>({ seq, dModel });
		// Each id selects one row of table; nothing is computed, only copied.
		for (std::size_t i = 0; i < seq; i++) {
			std::size_t idx = x[i];
			// An id past the table would read outside it.
			validation::requireBelow(idx, vocabSize, "Token id");
			// Consumes a sequence of IDs and produces a sequence of vectors.
			// logic: look-up row x[i] in table and copy it to out[i].
			for (std::size_t j = 0; j < dModel; j++) {
				out.data[i * dModel + j] = table.value.data[idx * dModel + j];
			}
		}
	}

	/**
	 * @brief Update gradients for specific words that were used in input.
	 *
	 * Network will tell us vector for 'word' is slightly wrong, move it this way.
	 * dOut is feedback received from network; and, will add grads for related rows
	 * of table.grad matrix. It leaves all other rows as 0 because others did not involve.
	 *
	 * @param x Token ids passed to matching forward() call.
	 * @param dOut Gradient w.r.t. forward()'s output [seq, dModel]; row i
	 * belongs to token x[i].
	 */
	void backward(const std::vector<std::size_t>& x, const Tensor<T>& dOut) {
		validation::requireShape(dOut, x.size(), dModel, "Embedding output gradient");
		std::size_t seq = x.size();

		for (std::size_t i = 0; i < seq; i++) {
			std::size_t idx = x[i];
			validation::requireBelow(idx, vocabSize, "Token id");
			for (std::size_t j = 0; j < dModel; j++) {
				// += is there because a word might appear multiple times in same batch or sequence.
				// Exm: "the cat ate mouse"
				// "the" appears two times at index 0 and index 3.
				// if we used = instead of +=, we would overrite "the" at index 0.
				// We accumulate gradiensts for "the" = 0.1 + 0.2 = 0.3
				table.grad.data[idx * dModel + j] += dOut.data[i * dModel + j];
			}
		}
	}

	/**
	 * @brief Apply accumulated gradients to table.
	 * Steps:
	 * 1. Init by zeroGrad (clear history)
	 * 2. forward()
	 * 3. backward() (accumulate current gradients)
	 * 4. update() (Apply changes)
	 *
	 * @param lr Learning rate.
	 * @param rule Optimizer to use (plain SGD by default, or Adam with its
	 * timestep); see UpdateRule.
	 */
	void update(T lr, UpdateRule rule = UpdateRule::sgd()) {
		this->table.update(lr, rule);
	}

	/**
	 * @brief zeroGrad() spread over several threads (the table has a row for
	 * every word, so for a large vocabulary clearing it takes a while).
	 *
	 * @param threads Most threads to use; 1 is the same as zeroGrad().
	 */
	void zeroGradParallel(std::size_t threads) {
		table.zeroGradParallel(threads);
	}

	/**
	 * @brief update() spread over several threads; bit-identical result.
	 *
	 * @param threads Most threads to use; 1 is the same as update().
	 * @see Parameter::updateParallel
	 */
	void updateParallel(T lr, UpdateRule rule, std::size_t threads) {
		table.updateParallel(lr, rule, threads);
	}
};