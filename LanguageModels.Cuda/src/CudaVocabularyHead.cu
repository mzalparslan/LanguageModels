// The GPU implementation of CudaVocabularyHead.h. Compiled by nvcc (see
// nvcc-build.cmd); only built when the CUDA Toolkit is installed.
//
// This file holds the buffers, the launches and the public class. The kernels are in
// detail/VocabularyKernels.cuh, and the helpers they use in detail/.

#include "CudaVocabularyHead.h"

#include "CudaRuntime.h"
#include "Validation.h"
#include "detail/CudaHost.cuh"
#include "detail/DeviceArray.cuh"
#include "detail/DeviceMath.cuh"
#include "detail/VocabularyKernels.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace cuda {

	using namespace detail;

	// ====================================================================== Impl

	template <typename T>
	class VocabularyHead<T>::Impl {
	public:
		/**
		 * @brief One parameter with its gradient and Adam's two moment estimates.
		 */
		class Parameter {
		public:
			std::size_t count = 0;
			DeviceArray<T> value;
			DeviceArray<T> grad;
			DeviceArray<T> first;
			DeviceArray<T> second;
			bool hasMoments = false;

			void allocate(std::size_t size) {
				count = size;
				value.allocate(size);
				grad.allocate(size);
				grad.zero(size);
			}

			// Moments are only allocated when Adam is first used, so plain SGD never pays for them.
			void ensureMoments() {
				if (!hasMoments) {
					first.allocate(count);
					second.allocate(count);
					// Set the flag first: resetMoments() does nothing until moments exist, and
					// memory from cudaMalloc is not guaranteed to be zero.
					hasMoments = true;
					resetMoments();
				}
			}

			void resetMoments() {
				if (hasMoments) {
					first.zero(count);
					second.zero(count);
				}
			}
		};

		Impl(std::size_t sourceVocabSize, std::size_t targetVocabSize, std::size_t modelWidth)
			: sourceVocab(sourceVocabSize), targetVocab(targetVocabSize), width(modelWidth) {
			validation::requirePositiveSize(sourceVocabSize, "Source vocabulary size");
			validation::requirePositiveSize(targetVocabSize, "Target vocabulary size");
			validation::requirePositiveSize(modelWidth, "Model width");
			if (!isAvailable()) {
				throw CudaError("cuda::VocabularyHead: No CUDA device is available.");
			}

			sourceTable.allocate(sourceVocabSize * modelWidth);
			targetTable.allocate(targetVocabSize * modelWidth);
			projection.allocate(modelWidth * targetVocabSize);
			bias.allocate(targetVocabSize);
			flag.allocate(1);
		}

		void upload(const std::vector<T>& sourceValues, const std::vector<T>& targetValues,
			const std::vector<T>& projectionValues, const std::vector<T>& biasValues) {
			validation::requireSameSize(sourceValues.size(), sourceTable.count, "Source table");
			validation::requireSameSize(targetValues.size(), targetTable.count, "Target table");
			validation::requireSameSize(projectionValues.size(), projection.count, "Projection weights");
			validation::requireSameSize(biasValues.size(), bias.count, "Projection bias");

			copyToDevice(sourceTable, sourceValues);
			copyToDevice(targetTable, targetValues);
			copyToDevice(projection, projectionValues);
			copyToDevice(bias, biasValues);
			for (Parameter* parameter : parameters()) {
				parameter->resetMoments();
			}
		}

		void download(std::vector<T>& sourceValues, std::vector<T>& targetValues,
			std::vector<T>& projectionValues, std::vector<T>& biasValues) const {
			copyToHost(sourceTable, sourceValues);
			copyToHost(targetTable, targetValues);
			copyToHost(projection, projectionValues);
			copyToHost(bias, biasValues);
		}

		void zeroGradients() {
			for (Parameter* parameter : parameters()) {
				parameter->grad.zero(parameter->count);
			}
		}

		void lookup(const Parameter& table, std::size_t vocab, const std::vector<std::size_t>& idList,
			std::vector<T>& out) {
			validation::requireNonEmpty(idList.size(), "Token id sequence");
			const std::size_t count = idList.size();
			uploadIds(idList, vocab, "Token id");

			rows.ensure(count * width);
			gatherRowsKernel<T><<<blocksFor(count * width), blockSize>>>(table.value.get(), ids.get(), rows.get(), count, width);
			check(cudaGetLastError(), "gatherRows kernel launch");

			out.resize(count * width);
			check(cudaMemcpy(out.data(), rows.get(), out.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy (embeddings to host)");
		}

		void accumulate(Parameter& table, std::size_t vocab, const std::vector<std::size_t>& idList,
			const std::vector<T>& rowGradients) {
			validation::requireSameSize(rowGradients.size(), idList.size() * width, "Embedding gradient");
			if (idList.empty()) {
				return;
			}
			uploadIds(idList, vocab, "Token id");

			rows.ensure(rowGradients.size());
			check(cudaMemcpy(rows.get(), rowGradients.data(), rowGradients.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy (gradients to device)");
			scatterAddRowsKernel<T><<<1, blockSize>>>(table.grad.get(), ids.get(), rows.get(), idList.size(), width);
			check(cudaGetLastError(), "scatterAddRows kernel launch");
		}

		T outputLayer(const std::vector<T>& decoderOutput, const std::vector<std::size_t>& labelList,
			std::vector<T>& gradDecoderOutput) {
			validation::requireNonEmpty(labelList.size(), "Label sequence");
			const std::size_t count = labelList.size();
			validation::requireSameSize(decoderOutput.size(), count * width, "Decoder output");

			// Labels go to the GPU as ints (the ids buffer is free for other uses between calls).
			uploadIds(labelList, targetVocab, "Label token id");

			input.ensure(decoderOutput.size());
			check(cudaMemcpy(input.get(), decoderOutput.data(), decoderOutput.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy (decoder output to device)");

			// Scores of every word at every position, then the loss and its gradient with respect to them.
			logits.ensure(count * targetVocab);
			const dim3 projectGrid(blocksFor(targetVocab), static_cast<unsigned>(count));
			projectKernel<T><<<projectGrid, blockSize>>>(input.get(), projection.value.get(), bias.value.get(), logits.get(), width, targetVocab);
			check(cudaGetLastError(), "project kernel launch");

			rowLoss.ensure(count);
			softmaxCrossEntropyKernel<T><<<static_cast<unsigned>(count), blockSize>>>(logits.get(), ids.get(), rowLoss.get(), targetVocab);
			check(cudaGetLastError(), "softmaxCrossEntropy kernel launch");

			// Sum the loss in position order, as the CPU does. A loss that is not finite
			// poisons every gradient, so stop before computing them.
			std::vector<T> losses(count);
			check(cudaMemcpy(losses.data(), rowLoss.get(), count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy (loss to host)");
			T loss = T(0);
			for (std::size_t r = 0; r < count; r++) {
				validation::requireFinite(losses[r], "Training loss");
				loss += losses[r];
			}
			validation::requireFinite(loss, "Training loss");

			const dim3 gradientGrid(blocksFor(targetVocab), static_cast<unsigned>(width));
			projectionGradientKernel<T><<<gradientGrid, blockSize>>>(input.get(), logits.get(), projection.grad.get(), bias.grad.get(), count, width, targetVocab);
			check(cudaGetLastError(), "projectionGradient kernel launch");

			inputGradient.ensure(count * width);
			inputGradientKernel<T><<<static_cast<unsigned>(count * width), blockSize>>>(logits.get(), projection.value.get(), inputGradient.get(), width, targetVocab);
			check(cudaGetLastError(), "inputGradient kernel launch");

			gradDecoderOutput.resize(count * width);
			check(cudaMemcpy(gradDecoderOutput.data(), inputGradient.get(), gradDecoderOutput.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy (gradient to host)");
			return loss;
		}

		void updateSgd(T learningRate) {
			validation::requirePositiveFinite(learningRate, "Learning rate");
			clearFlag();
			for (Parameter* parameter : parameters()) {
				sgdKernel<T><<<blocksFor(parameter->count), blockSize>>>(parameter->value.get(), parameter->grad.get(), parameter->count, learningRate, flag.get());
				check(cudaGetLastError(), "sgd kernel launch");
			}
			throwIfNotFinite();
		}

		void updateAdam(T learningRate, std::size_t step) {
			validation::requirePositiveFinite(learningRate, "Learning rate");
			if (step == 0) {
				throw InvalidParameterError("Adam timestep must be >= 1!");
			}
			const double beta1 = 0.9;
			const double beta2 = 0.999;
			const double epsilon = 1e-8;
			// Bias-correction denominators depend only on the timestep.
			const double biasCorrection1 = 1.0 - std::pow(beta1, static_cast<double>(step));
			const double biasCorrection2 = 1.0 - std::pow(beta2, static_cast<double>(step));
			validation::requireNonZeroDenominator(biasCorrection1, "Adam first-moment bias correction");
			validation::requireNonZeroDenominator(biasCorrection2, "Adam second-moment bias correction");

			clearFlag();
			for (Parameter* parameter : parameters()) {
				parameter->ensureMoments();
				adamKernel<T><<<blocksFor(parameter->count), blockSize>>>(parameter->value.get(), parameter->grad.get(),
					parameter->first.get(), parameter->second.get(), parameter->count, learningRate,
					static_cast<T>(beta1), static_cast<T>(beta2), static_cast<T>(epsilon),
					static_cast<T>(biasCorrection1), static_cast<T>(biasCorrection2), flag.get());
				check(cudaGetLastError(), "adam kernel launch");
			}
			throwIfNotFinite();
		}

		std::size_t sourceVocab;
		std::size_t targetVocab;
		std::size_t width;

		Parameter sourceTable;
		Parameter targetTable;
		Parameter projection;
		Parameter bias;

	private:
		DeviceArray<int> ids;
		DeviceArray<int> flag;
		DeviceArray<T> rows;
		DeviceArray<T> input;
		DeviceArray<T> logits;
		DeviceArray<T> inputGradient;
		DeviceArray<T> rowLoss;

		std::array<Parameter*, 4> parameters() {
			return { &sourceTable, &targetTable, &projection, &bias };
		}
		std::array<const Parameter*, 4> parameters() const {
			return { &sourceTable, &targetTable, &projection, &bias };
		}

		static void copyToDevice(Parameter& parameter, const std::vector<T>& values) {
			check(cudaMemcpy(parameter.value.get(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy (weights to device)");
		}

		static void copyToHost(const Parameter& parameter, std::vector<T>& values) {
			values.resize(parameter.count);
			check(cudaMemcpy(values.data(), parameter.value.get(), values.size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy (weights to host)");
		}

		// Checks every id against the vocabulary and copies them to the GPU as ints.
		void uploadIds(const std::vector<std::size_t>& idList, std::size_t vocab, const char* what) {
			std::vector<int> converted(idList.size());
			for (std::size_t i = 0; i < idList.size(); i++) {
				validation::requireBelow(idList[i], vocab, what);
				converted[i] = static_cast<int>(idList[i]);
			}
			ids.ensure(converted.size());
			check(cudaMemcpy(ids.get(), converted.data(), converted.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy (ids to device)");
		}

		void clearFlag() {
			flag.zero(1);
		}

		// Reads the flag the optimizer kernels set and reports what they found.
		void throwIfNotFinite() {
			int found = 0;
			check(cudaMemcpy(&found, flag.get(), sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy (flag to host)");
			if (found & flagNaN) {
				throw NaNError("A gradient or an updated weight is NaN!");
			}
			if (found & flagInfinite) {
				throw NonFiniteError("A gradient or an updated weight is infinite!");
			}
		}
	};

	// ================================================================ the class

	template <typename T>
	VocabularyHead<T>::VocabularyHead(std::size_t sourceVocabSize, std::size_t targetVocabSize, std::size_t modelWidth)
		: impl(new Impl(sourceVocabSize, targetVocabSize, modelWidth)) {}

	template <typename T>
	VocabularyHead<T>::~VocabularyHead() = default;

	template <typename T>
	VocabularyHead<T>::VocabularyHead(VocabularyHead&&) noexcept = default;

	template <typename T>
	VocabularyHead<T>& VocabularyHead<T>::operator=(VocabularyHead&&) noexcept = default;

	template <typename T>
	std::size_t VocabularyHead<T>::sourceVocab() const { return impl->sourceVocab; }

	template <typename T>
	std::size_t VocabularyHead<T>::targetVocab() const { return impl->targetVocab; }

	template <typename T>
	std::size_t VocabularyHead<T>::width() const { return impl->width; }

	template <typename T>
	void VocabularyHead<T>::uploadWeights(const std::vector<T>& sourceTable, const std::vector<T>& targetTable,
		const std::vector<T>& projection, const std::vector<T>& projectionBias) {
		impl->upload(sourceTable, targetTable, projection, projectionBias);
	}

	template <typename T>
	void VocabularyHead<T>::downloadWeights(std::vector<T>& sourceTable, std::vector<T>& targetTable,
		std::vector<T>& projection, std::vector<T>& projectionBias) const {
		impl->download(sourceTable, targetTable, projection, projectionBias);
	}

	template <typename T>
	void VocabularyHead<T>::zeroGradients() { impl->zeroGradients(); }

	template <typename T>
	void VocabularyHead<T>::lookupSource(const std::vector<std::size_t>& ids, std::vector<T>& rows) const {
		impl->lookup(impl->sourceTable, impl->sourceVocab, ids, rows);
	}

	template <typename T>
	void VocabularyHead<T>::lookupTarget(const std::vector<std::size_t>& ids, std::vector<T>& rows) const {
		impl->lookup(impl->targetTable, impl->targetVocab, ids, rows);
	}

	template <typename T>
	T VocabularyHead<T>::outputLayer(const std::vector<T>& decoderOutput, const std::vector<std::size_t>& labels,
		std::vector<T>& gradDecoderOutput) {
		return impl->outputLayer(decoderOutput, labels, gradDecoderOutput);
	}

	template <typename T>
	void VocabularyHead<T>::accumulateSourceGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients) {
		impl->accumulate(impl->sourceTable, impl->sourceVocab, ids, rowGradients);
	}

	template <typename T>
	void VocabularyHead<T>::accumulateTargetGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients) {
		impl->accumulate(impl->targetTable, impl->targetVocab, ids, rowGradients);
	}

	template <typename T>
	void VocabularyHead<T>::updateSgd(T lr) { impl->updateSgd(lr); }

	template <typename T>
	void VocabularyHead<T>::updateAdam(T lr, std::size_t step) { impl->updateAdam(lr, step); }

	// The two precisions this project provides.
	template class VocabularyHead<float>;
	template class VocabularyHead<double>;

} // namespace cuda
