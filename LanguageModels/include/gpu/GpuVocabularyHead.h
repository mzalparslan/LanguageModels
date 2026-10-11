#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "CudaRuntime.h"
#include "CudaVocabularyHead.h"
#include "Exceptions.h"
#include "MetalRuntime.h"
#include "MetalVocabularyHead.h"

/**
 * @brief The GPU backend a model trains on, whichever vendor's it is.
 *
 * MiniTransformer and the pipelines use gpu::VocabularyHead and never name CUDA or
 * Metal: this header picks the backend that is present.
 *  - NVIDIA GPU with the CUDA Toolkit build: CUDA (float or double);
 *  - Apple GPU: Metal (float only, as Metal has no 64-bit floating point);
 *  - neither: not available, and gpu::VocabularyHead cannot be created.
 * When both exist, CUDA is used.
 */
namespace gpu {

	enum class Backend {
		None,
		Cuda,
		Metal
	};

	inline const char* toString(Backend backend) {
		switch (backend) {
		case Backend::None:
			return "None";
		case Backend::Cuda:
			return "CUDA";
		case Backend::Metal:
			return "Metal";
		}
		return "Unknown";
	}

	/**
	 * @brief The backend a head of element type T would use, or None. Never throws.
	 */
	template <typename T>
	Backend availableBackend() {
		if (cuda::isAvailable()) {
			return Backend::Cuda;
		}
		if constexpr (std::is_same_v<T, float>) {
			if (metal::isAvailable()) {
				return Backend::Metal;
			}
		}
		return Backend::None;
	}

	/**
	 * @brief Whether a GPU can be used for element type T. Never throws.
	 */
	template <typename T>
	bool isAvailable() {
		return availableBackend<T>() != Backend::None;
	}

	/**
	 * @brief The GPU a head of type T would use as one line of text, led by the backend's
	 * name, or a note saying why there is none. Never throws.
	 */
	template <typename T>
	std::string describeDevice() {
		switch (availableBackend<T>()) {
		case Backend::Cuda:
			return "CUDA: " + cuda::describeDevice();
		case Backend::Metal:
			return "Metal: " + metal::describeDevice();
		case Backend::None:
			break;
		}
		if constexpr (!std::is_same_v<T, float>) {
			if (metal::isAvailable()) {
				return "Metal is available, but it cannot compute in double: use a float model";
			}
		}
		return "no GPU is available (no CUDA device and no Metal device)";
	}

	/**
	 * @brief The vocabulary-sized part of a training step, on whichever GPU is present.
	 *
	 * Has the interface of cuda::VocabularyHead (see CudaVocabularyHead.h for what each
	 * member does) and forwards every call to the CUDA or the Metal head it holds.
	 *
	 * @tparam T float or double (double needs CUDA).
	 */
	template <typename T>
	class VocabularyHead {
	public:
		/**
		 * @throws InvalidParameterSizeError If a size is zero.
		 * @throws DeviceError If no GPU can be used for T (see describeDevice()), or the
		 * GPU runs out of memory.
		 */
		VocabularyHead(std::size_t sourceVocab, std::size_t targetVocab, std::size_t width)
			: head(make(sourceVocab, targetVocab, width)) {}

		/**
		 * @brief The backend this head runs on.
		 */
		Backend backend() const {
			return std::holds_alternative<cuda::VocabularyHead<T>>(head) ? Backend::Cuda : Backend::Metal;
		}

		std::size_t sourceVocab() const { return std::visit([](const auto& h) { return h.sourceVocab(); }, head); }
		std::size_t targetVocab() const { return std::visit([](const auto& h) { return h.targetVocab(); }, head); }
		std::size_t width() const { return std::visit([](const auto& h) { return h.width(); }, head); }

		void uploadWeights(const std::vector<T>& sourceTable, const std::vector<T>& targetTable,
			const std::vector<T>& projection, const std::vector<T>& projectionBias) {
			std::visit([&](auto& h) { h.uploadWeights(sourceTable, targetTable, projection, projectionBias); }, head);
		}

		void downloadWeights(std::vector<T>& sourceTable, std::vector<T>& targetTable,
			std::vector<T>& projection, std::vector<T>& projectionBias) const {
			std::visit([&](const auto& h) { h.downloadWeights(sourceTable, targetTable, projection, projectionBias); }, head);
		}

		void zeroGradients() {
			std::visit([](auto& h) { h.zeroGradients(); }, head);
		}

		void lookupSource(const std::vector<std::size_t>& ids, std::vector<T>& rows) const {
			std::visit([&](const auto& h) { h.lookupSource(ids, rows); }, head);
		}

		void lookupTarget(const std::vector<std::size_t>& ids, std::vector<T>& rows) const {
			std::visit([&](const auto& h) { h.lookupTarget(ids, rows); }, head);
		}

		T outputLayer(const std::vector<T>& decoderOutput, const std::vector<std::size_t>& labels,
			std::vector<T>& gradDecoderOutput) {
			return std::visit([&](auto& h) { return h.outputLayer(decoderOutput, labels, gradDecoderOutput); }, head);
		}

		void accumulateSourceGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients) {
			std::visit([&](auto& h) { h.accumulateSourceGradient(ids, rowGradients); }, head);
		}

		void accumulateTargetGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients) {
			std::visit([&](auto& h) { h.accumulateTargetGradient(ids, rowGradients); }, head);
		}

		void updateSgd(T lr) {
			std::visit([&](auto& h) { h.updateSgd(lr); }, head);
		}

		void updateAdam(T lr, std::size_t step) {
			std::visit([&](auto& h) { h.updateAdam(lr, step); }, head);
		}

	private:
		using Head = std::variant<cuda::VocabularyHead<T>, metal::VocabularyHead<T>>;
		Head head;

		static Head make(std::size_t sourceVocab, std::size_t targetVocab, std::size_t width) {
			switch (availableBackend<T>()) {
			case Backend::Cuda:
				return Head(std::in_place_type<cuda::VocabularyHead<T>>, sourceVocab, targetVocab, width);
			case Backend::Metal:
				return Head(std::in_place_type<metal::VocabularyHead<T>>, sourceVocab, targetVocab, width);
			case Backend::None:
				break;
			}
			throw DeviceError("gpu::VocabularyHead: " + describeDevice<T>() + ".");
		}
	};

} // namespace gpu
