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
	Parallel
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
	}
	return "Unknown";
}
