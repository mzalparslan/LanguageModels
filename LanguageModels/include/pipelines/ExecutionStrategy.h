#pragma once

/**
 * @brief How a pipeline runs training of a model.
 *
 * More strategies can be added later. Today only MiniTransformer acts on it;
 * other models accept strategy and train same way whichever is
 * chosen.
 */
enum class ExecutionStrategy {
	// One thread: MiniTransformer::trainStep().
	Sequential,

	// Several threads: MiniTransformer::trainStepMultipleThread(). Same
	// results as Sequential, only faster on a large vocabulary.
	Parallel,

	// The GPU: MiniTransformer::trainStepDevice(). The vocabulary-sized part of
	// every step runs on the GPU, so it is much faster on a large vocabulary.
	// Results are the same as Sequential to within rounding, not bit for bit.
	// Which GPU is up to the machine: an NVIDIA GPU through CUDA, or an Apple GPU
	// through Metal (float models only: Metal has no double). Training with it throws
	// DeviceError when no GPU can be used.
	Device
};

/**
 * @brief Name of a strategy, for logging.
 */
inline const char* toString(ExecutionStrategy strategy) {
	switch (strategy) {
	case ExecutionStrategy::Sequential:
		return "Sequential";
	case ExecutionStrategy::Parallel:
		return "Parallel";
	case ExecutionStrategy::Device:
		return "Device";
	}
	return "Unknown";
}
