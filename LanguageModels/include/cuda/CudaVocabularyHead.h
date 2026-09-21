#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "Exceptions.h"

namespace cuda {

	/**
	 * @brief The vocabulary-sized part of MiniTransformer's training step, kept on
	 * the GPU: the two embedding tables and the output projection, with their
	 * gradients and optimizer state.
	 *
	 * Almost all of a training step's time goes into work that grows with the
	 * vocabulary, not with the model: choosing the output word among tens of
	 * thousands, and updating three tables of millions of numbers when a sentence
	 * touches a few dozen of them. Those parts are what this class runs on the GPU.
	 * The small encoder and decoder layers (width 32, a few tokens) stay on the CPU,
	 * so per step only small arrays cross to the GPU and back: token ids, a few
	 * embedding rows, the decoder output and its gradient (a few kilobytes).
	 *
	 * The weights live here while training runs. Upload them before the first step and
	 * download them when training is done (see uploadWeights() and downloadWeights()).
	 * Adam's moment estimates are kept here as well and are not downloaded.
	 *
	 * Everything is row-major, as in the CPU model: a table [vocab, width] holds one
	 * word per row, and the projection is [width, targetVocab].
	 *
	 * The class is a template so it can run in float (fast, and what a GPU is built
	 * for) or in double (to match the CPU model to within rounding). It is defined in
	 * the LanguageModels.Cuda project for float and double only. Without the CUDA
	 * Toolkit the constructor throws CudaError.
	 *
	 * @tparam T float or double.
	 */
	template <typename T>
	class VocabularyHead {
	public:
		/**
		 * @param sourceVocab Rows of the source (encoder) embedding table.
		 * @param targetVocab Rows of the target (decoder) embedding table; also the
		 * number of output words.
		 * @param width Model width: the size of one embedding.
		 * @throws InvalidParameterSizeError If a size is zero.
		 * @throws CudaError If CUDA is not available or the GPU runs out of memory.
		 */
		VocabularyHead(std::size_t sourceVocab, std::size_t targetVocab, std::size_t width);
		~VocabularyHead();

		VocabularyHead(VocabularyHead&&) noexcept;
		VocabularyHead& operator=(VocabularyHead&&) noexcept;
		VocabularyHead(const VocabularyHead&) = delete;
		VocabularyHead& operator=(const VocabularyHead&) = delete;

		std::size_t sourceVocab() const;
		std::size_t targetVocab() const;
		std::size_t width() const;

		/**
		 * @brief Copies the four parameters to the GPU and starts optimizer state over
		 * (Adam's moments are cleared).
		 *
		 * @param sourceTable [sourceVocab * width] values.
		 * @param targetTable [targetVocab * width] values.
		 * @param projection [width * targetVocab] values.
		 * @param projectionBias [targetVocab] values.
		 * @throws InvalidSizeError If a vector has the wrong length.
		 */
		void uploadWeights(const std::vector<T>& sourceTable, const std::vector<T>& targetTable,
			const std::vector<T>& projection, const std::vector<T>& projectionBias);

		/**
		 * @brief Copies the four parameters back to the host, resizing the vectors.
		 */
		void downloadWeights(std::vector<T>& sourceTable, std::vector<T>& targetTable,
			std::vector<T>& projection, std::vector<T>& projectionBias) const;

		/**
		 * @brief Clears all gradients. Call once at the start of every step: the
		 * backward calls below add into them.
		 */
		void zeroGradients();

		/**
		 * @brief Looks up the embeddings of source words.
		 *
		 * @param ids Word ids.
		 * @param rows Result [ids.size() * width], one embedding per id.
		 * @throws InvalidSizeError If ids is empty.
		 * @throws InvalidParameterError If an id is outside the vocabulary.
		 */
		void lookupSource(const std::vector<std::size_t>& ids, std::vector<T>& rows) const;

		/**
		 * @brief Looks up the embeddings of target words; see lookupSource().
		 */
		void lookupTarget(const std::vector<std::size_t>& ids, std::vector<T>& rows) const;

		/**
		 * @brief The output layer for one sentence, forward and backward in one call:
		 * scores every word of the target vocabulary at every position, computes the
		 * cross-entropy loss against the labels, and adds the gradients of the
		 * projection weights and bias.
		 *
		 * @param decoderOutput Decoder output [rows * width] (rows = labels.size()).
		 * @param labels The correct word id at each position.
		 * @param gradDecoderOutput Result [rows * width]: the gradient of the loss with
		 * respect to decoderOutput, to send back through the decoder.
		 * @return Cross-entropy loss summed over the positions.
		 * @throws InvalidSizeError If labels is empty or decoderOutput has the wrong length.
		 * @throws InvalidParameterError If a label is outside the vocabulary.
		 * @throws NaNError, NonFiniteError If the loss is not finite (the gradients are
		 * then not to be trusted, and no step should be taken).
		 */
		T outputLayer(const std::vector<T>& decoderOutput, const std::vector<std::size_t>& labels,
			std::vector<T>& gradDecoderOutput);

		/**
		 * @brief Adds the gradient of the loss with respect to source embeddings, row
		 * by row, into the source table's gradient. A word that occurs twice has its
		 * rows added in order.
		 *
		 * @param ids The source word ids the embeddings came from.
		 * @param rowGradients [ids.size() * width] gradients, one row per id.
		 * @throws InvalidSizeError If rowGradients has the wrong length.
		 * @throws InvalidParameterError If an id is outside the vocabulary.
		 */
		void accumulateSourceGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients);

		/**
		 * @brief Like accumulateSourceGradient(), for the target table.
		 */
		void accumulateTargetGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients);

		/**
		 * @brief Plain gradient descent on all four parameters: value -= lr * clip(grad),
		 * with each gradient clipped to [-1, 1], as Parameter::update() does.
		 *
		 * @throws InvalidParameterError If lr is not positive.
		 * @throws NaNError, NonFiniteError If a gradient or an updated weight is not finite.
		 */
		void updateSgd(T lr);

		/**
		 * @brief One Adam step on all four parameters (with the same clipping,
		 * betas 0.9 and 0.999 and epsilon 1e-8 as Parameter::update()).
		 *
		 * @param lr Learning rate.
		 * @param step Adam's timestep, 1 for the first step and increasing after that.
		 * @throws InvalidParameterError If lr is not positive or step is 0.
		 * @throws NaNError, NonFiniteError If a gradient or an updated weight is not finite.
		 */
		void updateAdam(T lr, std::size_t step);

	private:
		class Impl;
		std::unique_ptr<Impl> impl;
	};

} // namespace cuda
