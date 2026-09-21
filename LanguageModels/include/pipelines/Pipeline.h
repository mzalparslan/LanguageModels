#pragma once

#include <concepts>
#include <utility>
#include <vector>

#include "Exceptions.h"
#include "ExecutionStrategy.h"
#include "Logger.h"
#include "ScopedBenchmarkTimer.h"

/**
 * @brief What it takes to train and evaluate one kind of model in a Pipeline.
 *
 * A model needs more than train() and evaluate() to be useful in a pipeline:
 * its word or character vocabulary comes from training data, so model
 * cannot even be built before data is seen; its samples differ (sentence
 * pairs, characters, sentences); and so do its results (BLEU for translation,
 * perplexity for a language model). A ModelAdapter specialization hides all of
 * that behind one interface, so Pipeline itself is same for every model.
 *
 * To make a new model usable, specialize this template, satisfying
 * PipelineAdapter, in its own header (see MiniTransformerPipeline.h).
 *
 * @tparam T Floating-point type of model.
 * @tparam Model model class, e.g. MiniTransformer<T>.
 * @tparam Parameters Settings of model and of its training.
 */
template <typename T, typename Model, typename Parameters>
class ModelAdapter;

/**
 * @brief interface every ModelAdapter provides to Pipeline.
 *
 * Adapter is built from its Parameters, is trained on a list of Samples
 * (building its model and vocabularies from them) and evaluated on another
 * list, returning its own kind of Result.
 */
template <typename Adapter>
concept PipelineAdapter = requires(Adapter adapter,
	const std::vector<typename Adapter::Sample>& samples,
	ExecutionStrategy strategy, Logger& logger,
	const typename Adapter::Parameters& parameters) {
		typename Adapter::Sample;
		typename Adapter::Parameters;
		typename Adapter::Result;
		requires std::constructible_from<Adapter, typename Adapter::Parameters>;
		{ adapter.train(samples, strategy, logger) } -> std::same_as<void>;
		{ adapter.evaluate(samples, logger) } -> std::same_as<typename Adapter::Result>;
};

/**
 * @brief Trains and evaluates a model on a dataset, same way whichever
 * model it is:
 *
 * @code
 * auto samples = DataLoader::loadSentencePairs("fra.txt");
 * auto dataSet = DataSplitter::trainTestSplit(std::move(samples));
 *
 * Pipeline<double, MiniTransformer<double>, TranslationParameters<double>>
 *     pipeline{ TranslationParameters<double>{}, ExecutionStrategy::Parallel, logger };
 * pipeline.train(dataSet.trainingData);
 * TranslationMetrics metrics = pipeline.evaluate(dataSet.testData);
 * @endcode
 *
 * @tparam T Floating-point type used for model's calculations.
 * @tparam Model model to train (MiniTransformer<T>, VanillaRNN<T>,
 * a DecoderOnlyModel, BertModel<T>).
 * @tparam Parameters Settings of model and its training; each model has
 * its own class (TranslationParameters, RnnParameters, GptParameters,
 * BertParameters).
 */
template <typename T, typename Model, typename Parameters>
	requires PipelineAdapter<ModelAdapter<T, Model, Parameters>>
class Pipeline {
public:
	using Adapter = ModelAdapter<T, Model, Parameters>;
	// One example of data this model learns from.
	using Sample = typename Adapter::Sample;
	// What evaluate() returns.
	using Result = typename Adapter::Result;

	/**
	 * @param parameters Settings of model and its training.
	 * @param strategy How training is run. Only MiniTransformer acts on it
	 * today; other models train same way whichever is chosen.
	 * @param logger Where progress is reported. Defaults to a shared logger at
	 * LogLevel::Info if caller doesn't supply one; it must outlive pipeline.
	 */
	explicit Pipeline(Parameters parameters,
		ExecutionStrategy strategy = ExecutionStrategy::Sequential,
		Logger& logger = Logger::instance())
		: adapter(std::move(parameters)), strategy(strategy), logger(logger) {}

	/**
	 * @brief Builds model's vocabulary from training data and trains
	 * model on it. Training again starts from a new model.
	 *
	 * @param trainingData Samples to learn from.
	 * @throws InvalidSizeError If trainingData is empty, or holds no usable sample.
	 * @throws Whatever model throws while training (for example NaNError
	 * if loss stops being finite).
	 */
	void train(const std::vector<Sample>& trainingData) {
		ScopedBenchmarkTimer benchmark(logger, "Pipeline::train");
		// Prevent evaluation if any stage of a retraining fails.
		trained = false;

		if (trainingData.empty()) {
			throw InvalidSizeError("Pipeline::train: Training data is empty!");
		}

		logger.info() << "Training started. Samples: " << trainingData.size()
			<< ", execution: " << toString(strategy);

		adapter.train(trainingData, strategy, logger);

		logger.info() << "Training completed.";
		trained = true;
	}

	/**
	 * @brief Scores trained model on held-out data.
	 *
	 * @param testData Samples model was not trained on.
	 * @return model's metrics, of kind its adapter defines.
	 * @throws PipelineStateError If train() has not completed.
	 * @throws InvalidSizeError If testData is empty, or holds no usable sample.
	 */
	[[nodiscard]]
	Result evaluate(const std::vector<Sample>& testData) {
		ScopedBenchmarkTimer benchmark(logger, "Pipeline::evaluate");

		if (!trained) {
			throw PipelineStateError("Pipeline::evaluate: Call train() first.");
		}
		if (testData.empty()) {
			throw InvalidSizeError("Pipeline::evaluate: Test data is empty!");
		}

		return adapter.evaluate(testData, logger);
	}

	/**
	 * @brief Whether train() has completed.
	 */
	bool isTrained() const { return trained; }

	/**
	 * @brief strategy training runs with.
	 */
	ExecutionStrategy executionStrategy() const { return strategy; }

	/**
	 * @brief model's adapter, for model-specific extras such as translating
	 * one sentence or generating text.
	 */
	Adapter& modelAdapter() { return adapter; }
	const Adapter& modelAdapter() const { return adapter; }

private:
	Adapter adapter;
	ExecutionStrategy strategy;
	Logger& logger;
	bool trained = false;
};
