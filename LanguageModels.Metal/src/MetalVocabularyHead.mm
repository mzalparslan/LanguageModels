// The Apple GPU implementation of MetalVocabularyHead.h. Objective-C++: compiled with
// -fobjc-arc and linked with the Metal and Foundation frameworks (see the Makefile).
// Off Apple platforms this file is not built; the header has a stand-in instead.
//
// The structure is that of the CUDA version (LanguageModels.Cuda/src/CudaVocabularyHead.cu):
// the same Impl, the same order of operations, the same errors. What differs is Metal's:
//  - float only: Metal has no 64-bit floating point;
//  - buffers are in shared memory (the CPU and the GPU see the same bytes), so uploading
//    and downloading are memory copies, done while no command buffer is running;
//  - every GPU pass is committed and waited for before the CPU touches a buffer again,
//    which is what keeps those copies safe (CUDA's stream order does that for it);
//  - the kernels are compiled from source when the first head is created.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalVocabularyHead.h"

#include "MetalRuntime.h"
#include "Validation.h"
#include "VocabularyKernelsSource.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace metal {

	namespace {

		// Threads per threadgroup of every kernel: the kernels' reductions are written for it.
		constexpr NSUInteger threadgroupSize = 256;

		// Bits a kernel sets in the error flag when it meets a value that is not finite.
		constexpr int flagNaN = 1;
		constexpr int flagInfinite = 2;

		std::string describe(NSError* error) {
			if (error == nil) {
				return "unknown error";
			}
			return std::string([[error localizedDescription] UTF8String]);
		}

		/**
		 * @brief The device, the command queue and the compiled kernels, made once.
		 */
		class Context {
		public:
			id<MTLDevice> device = nil;
			id<MTLCommandQueue> queue = nil;
			id<MTLComputePipelineState> gatherRows = nil;
			id<MTLComputePipelineState> scatterAddRows = nil;
			id<MTLComputePipelineState> project = nil;
			id<MTLComputePipelineState> softmaxCrossEntropy = nil;
			id<MTLComputePipelineState> projectionGradient = nil;
			id<MTLComputePipelineState> inputGradient = nil;
			id<MTLComputePipelineState> sgd = nil;
			id<MTLComputePipelineState> adam = nil;
		};

		id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
			id<MTLFunction> function = [library newFunctionWithName:name];
			if (function == nil) {
				throw MetalError(std::string("Metal kernel not found: ") + [name UTF8String]);
			}
			NSError* error = nil;
			id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
			if (state == nil) {
				throw MetalError(std::string("Creating the Metal pipeline ") + [name UTF8String] + " failed: " + describe(error));
			}
			if ([state maxTotalThreadsPerThreadgroup] < threadgroupSize) {
				throw MetalError(std::string("The Metal kernel ") + [name UTF8String] + " cannot run with "
					+ std::to_string(threadgroupSize) + " threads per threadgroup on this device.");
			}
			return state;
		}

		Context buildContext() {
			Context context;
			context.device = MTLCreateSystemDefaultDevice();
			if (context.device == nil) {
				throw MetalError("metal::VocabularyHead: No Metal device is available.");
			}
			context.queue = [context.device newCommandQueue];
			if (context.queue == nil) {
				throw MetalError("metal::VocabularyHead: Could not create a Metal command queue.");
			}

			// Fast math would let the compiler skip NaN checks and use looser exp and log.
			MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
			options.languageVersion = MTLLanguageVersion2_4;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
			if (@available(macOS 15.0, *)) {
				options.mathMode = MTLMathModeSafe;
			}
			else {
				options.fastMathEnabled = NO;
			}
#else
			options.fastMathEnabled = NO;
#endif

			NSError* error = nil;
			NSString* source = [NSString stringWithUTF8String:detail::vocabularyKernelsSource()];
			id<MTLLibrary> library = [context.device newLibraryWithSource:source options:options error:&error];
			if (library == nil) {
				throw MetalError("Compiling the Metal kernels failed: " + describe(error));
			}

			context.gatherRows = makePipeline(context.device, library, @"gatherRowsKernel");
			context.scatterAddRows = makePipeline(context.device, library, @"scatterAddRowsKernel");
			context.project = makePipeline(context.device, library, @"projectKernel");
			context.softmaxCrossEntropy = makePipeline(context.device, library, @"softmaxCrossEntropyKernel");
			context.projectionGradient = makePipeline(context.device, library, @"projectionGradientKernel");
			context.inputGradient = makePipeline(context.device, library, @"inputGradientKernel");
			context.sgd = makePipeline(context.device, library, @"sgdKernel");
			context.adam = makePipeline(context.device, library, @"adamKernel");
			return context;
		}

		// A failed build is not remembered: the next head tries again.
		Context& context() {
			static Context instance = buildContext();
			return instance;
		}

		NSUInteger groupsFor(std::size_t count) {
			return static_cast<NSUInteger>((count + threadgroupSize - 1) / threadgroupSize);
		}

		/**
		 * @brief An array in shared memory, visible to the CPU and the GPU alike. Grows
		 * (discarding its contents) when asked for more than it holds.
		 */
		template <typename E>
		class DeviceArray {
		public:
			void allocate(std::size_t count) {
				buffer = [context().device newBufferWithLength:count * sizeof(E) options:MTLResourceStorageModeShared];
				if (buffer == nil) {
					throw MetalError("Metal could not allocate " + std::to_string(count * sizeof(E)) + " bytes.");
				}
				capacity = count;
			}

			void ensure(std::size_t count) {
				if (count > capacity) {
					allocate(count);
				}
			}

			// Only call while no GPU pass is running (every pass is waited for before it returns).
			void zero(std::size_t count) {
				std::memset([buffer contents], 0, count * sizeof(E));
			}

			E* data() const { return static_cast<E*>([buffer contents]); }
			id<MTLBuffer> get() const { return buffer; }

		private:
			id<MTLBuffer> buffer = nil;
			std::size_t capacity = 0;
		};

		/**
		 * @brief One command buffer with one compute encoder: encode kernels, then finish()
		 * to run them in order and wait for the result.
		 */
		class Pass {
		public:
			Pass() {
				commandBuffer = [context().queue commandBuffer];
				encoder = [commandBuffer computeCommandEncoder];
				if (commandBuffer == nil || encoder == nil) {
					throw MetalError("Metal could not create a command buffer.");
				}
			}

			// Metal aborts the program if an encoder is released without being ended, which
			// is what an exception between the first kernel and finish() would do.
			~Pass() {
				if (!finished) {
					[encoder endEncoding];
				}
			}

			Pass(const Pass&) = delete;
			Pass& operator=(const Pass&) = delete;

			void use(id<MTLComputePipelineState> pipeline) {
				[encoder setComputePipelineState:pipeline];
			}

			template <typename E>
			void buffer(const DeviceArray<E>& array, NSUInteger index) {
				[encoder setBuffer:array.get() offset:0 atIndex:index];
			}

			template <typename V>
			void value(V number, NSUInteger index) {
				[encoder setBytes:&number length:sizeof(V) atIndex:index];
			}

			void dispatch(NSUInteger groupsX, NSUInteger groupsY = 1) {
				[encoder dispatchThreadgroups:MTLSizeMake(groupsX, groupsY, 1)
					threadsPerThreadgroup:MTLSizeMake(threadgroupSize, 1, 1)];
			}

			void finish(const char* what) {
				[encoder endEncoding];
				finished = true;
				[commandBuffer commit];
				[commandBuffer waitUntilCompleted];
				if ([commandBuffer status] == MTLCommandBufferStatusError) {
					throw MetalError(std::string(what) + " failed: " + describe([commandBuffer error]));
				}
			}

		private:
			id<MTLCommandBuffer> commandBuffer = nil;
			id<MTLComputeCommandEncoder> encoder = nil;
			bool finished = false;
		};

		// Sizes and counts are 32-bit in the kernels.
		std::uint32_t asUint(std::size_t value) {
			return static_cast<std::uint32_t>(value);
		}

	} // namespace

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
					// a new buffer is not guaranteed to be zero.
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
			if constexpr (!std::is_same_v<T, float>) {
				throw MetalError("metal::VocabularyHead: Metal cannot compute in double; use a float model.");
			}
			if (!isAvailable()) {
				throw MetalError("metal::VocabularyHead: No Metal device is available.");
			}
			// The kernels index with 32-bit integers.
			const std::size_t largest = std::numeric_limits<std::uint32_t>::max() / 2;
			if (sourceVocabSize * modelWidth > largest || targetVocabSize * modelWidth > largest) {
				throw InvalidParameterError("metal::VocabularyHead: The tables are too large for Metal's 32-bit indexing.");
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
			Pass pass;
			pass.use(context().gatherRows);
			pass.buffer(table.value, 0);
			pass.buffer(ids, 1);
			pass.buffer(rows, 2);
			pass.value(asUint(count), 3);
			pass.value(asUint(width), 4);
			pass.dispatch(groupsFor(count * width));
			pass.finish("gatherRows kernel");

			out.assign(rows.data(), rows.data() + count * width);
		}

		void accumulate(Parameter& table, std::size_t vocab, const std::vector<std::size_t>& idList,
			const std::vector<T>& rowGradients) {
			validation::requireSameSize(rowGradients.size(), idList.size() * width, "Embedding gradient");
			if (idList.empty()) {
				return;
			}
			uploadIds(idList, vocab, "Token id");

			rows.ensure(rowGradients.size());
			std::memcpy(rows.data(), rowGradients.data(), rowGradients.size() * sizeof(T));
			Pass pass;
			pass.use(context().scatterAddRows);
			pass.buffer(table.grad, 0);
			pass.buffer(ids, 1);
			pass.buffer(rows, 2);
			pass.value(asUint(idList.size()), 3);
			pass.value(asUint(width), 4);
			pass.dispatch(1);
			pass.finish("scatterAddRows kernel");
		}

		T outputLayer(const std::vector<T>& decoderOutput, const std::vector<std::size_t>& labelList,
			std::vector<T>& gradDecoderOutput) {
			validation::requireNonEmpty(labelList.size(), "Label sequence");
			const std::size_t count = labelList.size();
			validation::requireSameSize(decoderOutput.size(), count * width, "Decoder output");

			// Labels go to the GPU as ints (the ids buffer is free for other uses between calls).
			uploadIds(labelList, targetVocab, "Label token id");

			input.ensure(decoderOutput.size());
			std::memcpy(input.data(), decoderOutput.data(), decoderOutput.size() * sizeof(T));

			// Scores of every word at every position, then the loss and its gradient with respect to them.
			logits.ensure(count * targetVocab);
			rowLoss.ensure(count);
			{
				Pass pass;
				pass.use(context().project);
				pass.buffer(input, 0);
				pass.buffer(projection.value, 1);
				pass.buffer(bias.value, 2);
				pass.buffer(logits, 3);
				pass.value(asUint(width), 4);
				pass.value(asUint(targetVocab), 5);
				pass.dispatch(groupsFor(targetVocab), count);

				pass.use(context().softmaxCrossEntropy);
				pass.buffer(logits, 0);
				pass.buffer(ids, 1);
				pass.buffer(rowLoss, 2);
				pass.value(asUint(targetVocab), 3);
				pass.dispatch(count);
				pass.finish("output layer");
			}

			// Sum the loss in position order, as the CPU does. A loss that is not finite
			// poisons every gradient, so stop before computing them.
			T loss = T(0);
			for (std::size_t r = 0; r < count; r++) {
				const T rowValue = rowLoss.data()[r];
				validation::requireFinite(rowValue, "Training loss");
				loss += rowValue;
			}
			validation::requireFinite(loss, "Training loss");

			inputGradient.ensure(count * width);
			{
				Pass pass;
				pass.use(context().projectionGradient);
				pass.buffer(input, 0);
				pass.buffer(logits, 1);
				pass.buffer(projection.grad, 2);
				pass.buffer(bias.grad, 3);
				pass.value(asUint(count), 4);
				pass.value(asUint(width), 5);
				pass.value(asUint(targetVocab), 6);
				pass.dispatch(groupsFor(targetVocab), width);

				pass.use(context().inputGradient);
				pass.buffer(logits, 0);
				pass.buffer(projection.value, 1);
				pass.buffer(inputGradient, 2);
				pass.value(asUint(width), 3);
				pass.value(asUint(targetVocab), 4);
				pass.dispatch(count * width);
				pass.finish("gradient of the output layer");
			}

			gradDecoderOutput.assign(inputGradient.data(), inputGradient.data() + count * width);
			return loss;
		}

		void updateSgd(T learningRate) {
			validation::requirePositiveFinite(learningRate, "Learning rate");
			clearFlag();
			Pass pass;
			for (Parameter* parameter : parameters()) {
				pass.use(context().sgd);
				pass.buffer(parameter->value, 0);
				pass.buffer(parameter->grad, 1);
				pass.value(asUint(parameter->count), 2);
				pass.value(static_cast<float>(learningRate), 3);
				pass.buffer(flag, 4);
				pass.dispatch(groupsFor(parameter->count));
			}
			pass.finish("sgd kernel");
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
			}
			Pass pass;
			for (Parameter* parameter : parameters()) {
				pass.use(context().adam);
				pass.buffer(parameter->value, 0);
				pass.buffer(parameter->grad, 1);
				pass.buffer(parameter->first, 2);
				pass.buffer(parameter->second, 3);
				pass.value(asUint(parameter->count), 4);
				pass.value(static_cast<float>(learningRate), 5);
				pass.value(static_cast<float>(beta1), 6);
				pass.value(static_cast<float>(beta2), 7);
				pass.value(static_cast<float>(epsilon), 8);
				pass.value(static_cast<float>(biasCorrection1), 9);
				pass.value(static_cast<float>(biasCorrection2), 10);
				pass.buffer(flag, 11);
				pass.dispatch(groupsFor(parameter->count));
			}
			pass.finish("adam kernel");
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
			std::memcpy(parameter.value.data(), values.data(), values.size() * sizeof(T));
		}

		static void copyToHost(const Parameter& parameter, std::vector<T>& values) {
			values.assign(parameter.value.data(), parameter.value.data() + parameter.count);
		}

		// Checks every id against the vocabulary and copies them to the GPU as ints.
		void uploadIds(const std::vector<std::size_t>& idList, std::size_t vocab, const char* what) {
			for (std::size_t id : idList) {
				validation::requireBelow(id, vocab, what);
			}
			ids.ensure(idList.size());
			int* destination = ids.data();
			for (std::size_t i = 0; i < idList.size(); i++) {
				destination[i] = static_cast<int>(idList[i]);
			}
		}

		void clearFlag() {
			flag.zero(1);
		}

		// Reads the flag the optimizer kernels set and reports what they found.
		void throwIfNotFinite() {
			const int found = flag.data()[0];
			if (found & flagNaN) {
				throw NaNError("A gradient or an updated weight is NaN!");
			}
			if (found & flagInfinite) {
				throw NonFiniteError("A gradient or an updated weight is infinite!");
			}
		}
	};

	// ================================================================ the class
	//
	// Every call that touches Metal runs inside an autorelease pool: the command buffers and
	// encoders it creates are autoreleased, and a training run makes hundreds of thousands
	// of them on a thread that has no pool of its own.

	template <typename T>
	VocabularyHead<T>::VocabularyHead(std::size_t sourceVocabSize, std::size_t targetVocabSize, std::size_t modelWidth) {
		@autoreleasepool {
			impl.reset(new Impl(sourceVocabSize, targetVocabSize, modelWidth));
		}
	}

	template <typename T>
	VocabularyHead<T>::~VocabularyHead() {
		@autoreleasepool {
			impl.reset();
		}
	}

	template <typename T>
	VocabularyHead<T>::VocabularyHead(VocabularyHead&&) noexcept = default;

	template <typename T>
	VocabularyHead<T>& VocabularyHead<T>::operator=(VocabularyHead&& other) noexcept {
		@autoreleasepool {
			impl = std::move(other.impl);
		}
		return *this;
	}

	template <typename T>
	std::size_t VocabularyHead<T>::sourceVocab() const { return impl->sourceVocab; }

	template <typename T>
	std::size_t VocabularyHead<T>::targetVocab() const { return impl->targetVocab; }

	template <typename T>
	std::size_t VocabularyHead<T>::width() const { return impl->width; }

	template <typename T>
	void VocabularyHead<T>::uploadWeights(const std::vector<T>& sourceTable, const std::vector<T>& targetTable,
		const std::vector<T>& projection, const std::vector<T>& projectionBias) {
		@autoreleasepool {
			impl->upload(sourceTable, targetTable, projection, projectionBias);
		}
	}

	template <typename T>
	void VocabularyHead<T>::downloadWeights(std::vector<T>& sourceTable, std::vector<T>& targetTable,
		std::vector<T>& projection, std::vector<T>& projectionBias) const {
		@autoreleasepool {
			impl->download(sourceTable, targetTable, projection, projectionBias);
		}
	}

	template <typename T>
	void VocabularyHead<T>::zeroGradients() {
		@autoreleasepool {
			impl->zeroGradients();
		}
	}

	template <typename T>
	void VocabularyHead<T>::lookupSource(const std::vector<std::size_t>& ids, std::vector<T>& rows) const {
		@autoreleasepool {
			impl->lookup(impl->sourceTable, impl->sourceVocab, ids, rows);
		}
	}

	template <typename T>
	void VocabularyHead<T>::lookupTarget(const std::vector<std::size_t>& ids, std::vector<T>& rows) const {
		@autoreleasepool {
			impl->lookup(impl->targetTable, impl->targetVocab, ids, rows);
		}
	}

	template <typename T>
	T VocabularyHead<T>::outputLayer(const std::vector<T>& decoderOutput, const std::vector<std::size_t>& labels,
		std::vector<T>& gradDecoderOutput) {
		@autoreleasepool {
			return impl->outputLayer(decoderOutput, labels, gradDecoderOutput);
		}
	}

	template <typename T>
	void VocabularyHead<T>::accumulateSourceGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients) {
		@autoreleasepool {
			impl->accumulate(impl->sourceTable, impl->sourceVocab, ids, rowGradients);
		}
	}

	template <typename T>
	void VocabularyHead<T>::accumulateTargetGradient(const std::vector<std::size_t>& ids, const std::vector<T>& rowGradients) {
		@autoreleasepool {
			impl->accumulate(impl->targetTable, impl->targetVocab, ids, rowGradients);
		}
	}

	template <typename T>
	void VocabularyHead<T>::updateSgd(T lr) {
		@autoreleasepool {
			impl->updateSgd(lr);
		}
	}

	template <typename T>
	void VocabularyHead<T>::updateAdam(T lr, std::size_t step) {
		@autoreleasepool {
			impl->updateAdam(lr, step);
		}
	}

	// float is what Metal computes in. double is instantiated only so that creating one
	// reports a clear error instead of failing to link.
	template class VocabularyHead<float>;
	template class VocabularyHead<double>;

} // namespace metal
