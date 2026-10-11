#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "Exceptions.h"

namespace metal {

	/**
	 * @brief The vocabulary-sized part of MiniTransformer's training step, kept on an
	 * Apple GPU through Metal: the two embedding tables and the output projection,
	 * with their gradients and optimizer state.
	 *
	 * It has the same interface and the same behavior as cuda::VocabularyHead (see
	 * CudaVocabularyHead.h for the full description of every member), so the two are
	 * interchangeable behind gpu::VocabularyHead. Two things differ:
	 *  - Metal has no 64-bit floating point, so only float can be used. Creating a
	 *    VocabularyHead<double> throws MetalError.
	 *  - The CPU and the GPU share memory on Apple Silicon, so "uploading" and
	 *    "downloading" are plain copies within one memory.
	 *
	 * On Apple platforms the class is implemented in LanguageModels.Metal. Elsewhere it
	 * is a stand-in at the bottom of this file: the constructor throws MetalError.
	 *
	 * @tparam T float (double is accepted by the compiler but throws when created).
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
		 * @throws MetalError If Metal is not available, T is double, or the GPU runs out of memory.
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

		void uploadWeights(const std::vector<T>& sourceTable, const std::vector<T>& targetTable,
			const std::vector<T>& projection, const std::vector<T>& projectionBias);

		void downloadWeights(std::vector<T>& sourceTable, std::vector<T>& targetTable,
			std::vector<T>& projection, std::vector<T>& projectionBias) const;

		void zeroGradients();

		void lookupSource(const std::vector<std::size_t>& ids, std::vector<T>& rows) const;
		void lookupTarget(const std::vector<std::size_t>& ids, std::vector<T>& rows) const;

		T outputLayer(const std::vector<T>& decoderOutput, const std::vector<std::size_t>& labels,
			std::vector<T>& gradDecoderOutput);

		void accumulateSourceGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients);
		void accumulateTargetGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients);

		void updateSgd(T lr);
		void updateAdam(T lr, std::size_t step);

	private:
		class Impl;
		std::unique_ptr<Impl> impl;
	};

#if !defined(__APPLE__)

	// ---------------------------------------------------------------- stand-in
	// Used on every platform without Metal. The head cannot be created, so callers must
	// check metal::isAvailable() first; every member throws so a mistake is reported.

	namespace detail {
		[[noreturn]] inline void unsupported() {
			throw MetalError("metal::VocabularyHead: Metal is only available on Apple platforms.");
		}
	}

	template <typename T>
	class VocabularyHead<T>::Impl {};

	template <typename T>
	VocabularyHead<T>::VocabularyHead(std::size_t, std::size_t, std::size_t) { detail::unsupported(); }

	template <typename T>
	VocabularyHead<T>::~VocabularyHead() = default;

	template <typename T>
	VocabularyHead<T>::VocabularyHead(VocabularyHead&&) noexcept = default;

	template <typename T>
	VocabularyHead<T>& VocabularyHead<T>::operator=(VocabularyHead&&) noexcept = default;

	template <typename T>
	std::size_t VocabularyHead<T>::sourceVocab() const { detail::unsupported(); }

	template <typename T>
	std::size_t VocabularyHead<T>::targetVocab() const { detail::unsupported(); }

	template <typename T>
	std::size_t VocabularyHead<T>::width() const { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::uploadWeights(const std::vector<T>&, const std::vector<T>&,
		const std::vector<T>&, const std::vector<T>&) { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::downloadWeights(std::vector<T>&, std::vector<T>&,
		std::vector<T>&, std::vector<T>&) const { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::zeroGradients() { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::lookupSource(const std::vector<std::size_t>&, std::vector<T>&) const { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::lookupTarget(const std::vector<std::size_t>&, std::vector<T>&) const { detail::unsupported(); }

	template <typename T>
	T VocabularyHead<T>::outputLayer(const std::vector<T>&, const std::vector<std::size_t>&, std::vector<T>&) { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::accumulateSourceGradient(const std::vector<std::size_t>&, const std::vector<T>&) { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::accumulateTargetGradient(const std::vector<std::size_t>&, const std::vector<T>&) { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::updateSgd(T) { detail::unsupported(); }

	template <typename T>
	void VocabularyHead<T>::updateAdam(T, std::size_t) { detail::unsupported(); }

#endif

} // namespace metal
