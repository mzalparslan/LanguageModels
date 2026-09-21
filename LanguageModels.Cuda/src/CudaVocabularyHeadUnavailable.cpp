// The implementation of CudaVocabularyHead.h used when the build has no CUDA Toolkit.
// The head cannot be created, so callers must check cuda::isAvailable() first; every
// member throws CudaError so that a mistake is reported instead of ignored.

#include "CudaVocabularyHead.h"

namespace cuda {

	namespace {
		[[noreturn]] void unsupported() {
			throw CudaError("cuda::VocabularyHead: This build has no CUDA support.");
		}
	}

	template <typename T>
	class VocabularyHead<T>::Impl {};

	template <typename T>
	VocabularyHead<T>::VocabularyHead(std::size_t, std::size_t, std::size_t) { unsupported(); }

	template <typename T>
	VocabularyHead<T>::~VocabularyHead() = default;

	template <typename T>
	VocabularyHead<T>::VocabularyHead(VocabularyHead&&) noexcept = default;

	template <typename T>
	VocabularyHead<T>& VocabularyHead<T>::operator=(VocabularyHead&&) noexcept = default;

	template <typename T>
	std::size_t VocabularyHead<T>::sourceVocab() const { unsupported(); }

	template <typename T>
	std::size_t VocabularyHead<T>::targetVocab() const { unsupported(); }

	template <typename T>
	std::size_t VocabularyHead<T>::width() const { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::uploadWeights(const std::vector<T>&, const std::vector<T>&,
		const std::vector<T>&, const std::vector<T>&) { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::downloadWeights(std::vector<T>&, std::vector<T>&,
		std::vector<T>&, std::vector<T>&) const { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::zeroGradients() { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::lookupSource(const std::vector<std::size_t>&, std::vector<T>&) const { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::lookupTarget(const std::vector<std::size_t>&, std::vector<T>&) const { unsupported(); }

	template <typename T>
	T VocabularyHead<T>::outputLayer(const std::vector<T>&, const std::vector<std::size_t>&, std::vector<T>&) { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::accumulateSourceGradient(const std::vector<std::size_t>&, const std::vector<T>&) { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::accumulateTargetGradient(const std::vector<std::size_t>&, const std::vector<T>&) { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::updateSgd(T) { unsupported(); }

	template <typename T>
	void VocabularyHead<T>::updateAdam(T, std::size_t) { unsupported(); }

	template class VocabularyHead<float>;
	template class VocabularyHead<double>;

} // namespace cuda
