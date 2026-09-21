#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <random>
#include <vector>

#include "Dataset.h"
#include "Exceptions.h"

/**
 * @brief Splits samples into a training part and a test part.
 *
 * Two ways, for two kinds of data:
 *  - trainTestSplit() shuffles first, for independent examples such as
 *    sentence pairs, so test set is a fair sample of whole file;
 *  - sequentialSplit() keeps order and cuts once, for text that is read
 *    as a stream (a language model, or BERT's "next sentence") where shuffling
 *    would tear it apart and let neighbouring text leak into test set.
 */
class DataSplitter {
	// Prevent instantiation of this class.
	DataSplitter() = delete;

public:
	/**
	 * @brief Randomly splits samples into training and test datasets.
	 *
	 * random seed makes shuffle reproducible.
	 *
	 * @param samples Samples to shuffle and split.
	 * @param trainingRatio Share assigned to training, strictly between 0 and 1
	 * (default 0.8: 80% training, 20% test).
	 * @param randomSeed Seed of shuffle.
	 * @throws InvalidParameterError If trainingRatio is not finite or not
	 * strictly between 0 and 1.
	 * @throws InvalidSizeError If samples is empty, or too few for both parts
	 * to get at least one sample at this ratio.
	 */
	template <typename Sample>
	[[nodiscard]]
	static Dataset<Sample> trainTestSplit(std::vector<Sample> samples,
		double trainingRatio = 0.8, std::uint32_t randomSeed = 42) {
		const std::size_t trainingSize = trainingSizeFor(samples.size(), trainingRatio);

		std::mt19937 randomGenerator(randomSeed);
		std::shuffle(samples.begin(), samples.end(), randomGenerator);

		return cutAt(std::move(samples), trainingSize);
	}

	/**
	 * @brief Splits samples at one position: first part is training and the
	 * rest is test, in their original order.
	 *
	 * @param samples Samples to split.
	 * @param trainingRatio Share assigned to training, strictly between 0 and 1
	 * (default 0.9).
	 * @throws InvalidParameterError See trainTestSplit().
	 * @throws InvalidSizeError See trainTestSplit().
	 */
	template <typename Sample>
	[[nodiscard]]
	static Dataset<Sample> sequentialSplit(std::vector<Sample> samples, double trainingRatio = 0.9) {
		const std::size_t trainingSize = trainingSizeFor(samples.size(), trainingRatio);
		return cutAt(std::move(samples), trainingSize);
	}

private:
	static std::size_t trainingSizeFor(std::size_t sampleCount, double trainingRatio) {
		if (!std::isfinite(trainingRatio)) {
			throw InvalidParameterError("DataSplitter: Training ratio is not finite!");
		}
		if (trainingRatio <= 0.0 || trainingRatio >= 1.0) {
			throw InvalidParameterError("DataSplitter: Training ratio must be between 0 and 1!");
		}
		if (sampleCount == 0) {
			throw InvalidSizeError("DataSplitter: Samples vector is empty!");
		}

		const std::size_t trainingSize =
			static_cast<std::size_t>(static_cast<double>(sampleCount) * trainingRatio);
		if (trainingSize == 0 || trainingSize >= sampleCount) {
			throw InvalidSizeError("DataSplitter: Too few samples to give both parts at least one!");
		}
		return trainingSize;
	}

	template <typename Sample>
	static Dataset<Sample> cutAt(std::vector<Sample> samples, std::size_t trainingSize) {
		const auto splitPosition = samples.begin() + static_cast<std::ptrdiff_t>(trainingSize);

		Dataset<Sample> dataset;
		dataset.trainingData.assign(std::make_move_iterator(samples.begin()),
			std::make_move_iterator(splitPosition));
		dataset.testData.assign(std::make_move_iterator(splitPosition),
			std::make_move_iterator(samples.end()));
		return dataset;
	}
};
