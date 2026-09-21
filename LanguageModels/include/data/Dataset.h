#pragma once

#include <vector>

/**
 * @brief A dataset split into part a model learns from and part it is
 * judged on.
 *
 * @tparam Sample One example: a SentencePair for translation, a character for
 * a character-level language model, a sentence for BERT.
 */
template <typename Sample>
class Dataset {
public:
	std::vector<Sample> trainingData;
	std::vector<Sample> testData;
};
