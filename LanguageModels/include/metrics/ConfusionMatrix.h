#pragma once

#include <cstddef>
#include <vector>

#include "Validation.h"

/**
 * @brief Confusion matrix for single-label classification (BERT's next-sentence
 * head, a masked-token classifier, ...) with scores derived from it.
 *
 * Rows are actual class, columns predicted class, so count(a, p) is
 * how many examples of class a were predicted as p. Precision, recall and F1
 * are defined as 0 when their denominator is zero (a class never predicted,
 * or never present), and macro averages include every class, as
 * scikit-learn does.
 */
class ConfusionMatrix {
public:
	/**
	 * @param numClasses Number of classes; ids are 0 .. numClasses - 1.
	 * @throws InvalidParameterSizeError If numClasses is zero.
	 */
	explicit ConfusionMatrix(std::size_t numClasses)
		: classCount(numClasses)
	{
		validation::requirePositiveSize(numClasses, "Class count");
		counts.assign(numClasses * numClasses, 0);
	}

	/**
	 * @brief Records one example.
	 *
	 * @throws InvalidParameterError If either class id is out of range.
	 */
	void add(std::size_t actual, std::size_t predicted) {
		validation::requireBelow(actual, classCount, "Actual class id");
		validation::requireBelow(predicted, classCount, "Predicted class id");
		counts[actual * classCount + predicted]++;
		totalCount++;
	}

	/**
	 * @brief Records many examples at once.
	 *
	 * @throws InvalidSizeError If vectors differ in length.
	 * @throws InvalidParameterError If a class id is out of range.
	 */
	void add(const std::vector<std::size_t>& actual, const std::vector<std::size_t>& predicted) {
		validation::requireSameSize(predicted.size(), actual.size(), "Predicted classes");
		for (std::size_t i = 0; i < actual.size(); i++) {
			add(actual[i], predicted[i]);
		}
	}

	/**
	 * @brief Number of classes.
	 */
	std::size_t numClasses() const { return classCount; }

	/**
	 * @brief Number of examples recorded.
	 */
	std::size_t total() const { return totalCount; }

	/**
	 * @brief Examples of class `actual` that were predicted as class `predicted`.
	 *
	 * @throws InvalidParameterError If either class id is out of range.
	 */
	std::size_t count(std::size_t actual, std::size_t predicted) const {
		validation::requireBelow(actual, classCount, "Actual class id");
		validation::requireBelow(predicted, classCount, "Predicted class id");
		return counts[actual * classCount + predicted];
	}

	/**
	 * @brief Fraction of examples predicted correctly.
	 *
	 * @throws DivisionByZeroError If no example was recorded.
	 */
	double accuracy() const {
		validation::requireNonZeroDenominator(totalCount, "Confusion matrix example count");
		std::size_t correct = 0;
		for (std::size_t c = 0; c < classCount; c++) {
			correct += counts[c * classCount + c];
		}
		return static_cast<double>(correct) / static_cast<double>(totalCount);
	}

	/**
	 * @brief TP / (TP + FP) for one class: of examples predicted as it,
	 * how many really were.
	 */
	double precision(std::size_t classId) const {
		validation::requireBelow(classId, classCount, "Class id");
		std::size_t predicted = 0;
		for (std::size_t actual = 0; actual < classCount; actual++) {
			predicted += counts[actual * classCount + classId];
		}
		return ratio(counts[classId * classCount + classId], predicted);
	}

	/**
	 * @brief TP / (TP + FN) for one class: of examples that really are
	 * it, how many were found.
	 */
	double recall(std::size_t classId) const {
		validation::requireBelow(classId, classCount, "Class id");
		std::size_t actual = 0;
		for (std::size_t predicted = 0; predicted < classCount; predicted++) {
			actual += counts[classId * classCount + predicted];
		}
		return ratio(counts[classId * classCount + classId], actual);
	}

	/**
	 * @brief Harmonic mean of precision and recall for one class.
	 */
	double f1(std::size_t classId) const {
		const double p = precision(classId);
		const double r = recall(classId);
		return (p + r == 0.0) ? 0.0 : 2.0 * p * r / (p + r);
	}

	/**
	 * @brief Unweighted mean of per-class precision over all classes.
	 */
	double macroPrecision() const { return macroAverage(&ConfusionMatrix::precision); }

	/**
	 * @brief Unweighted mean of per-class recall over all classes.
	 */
	double macroRecall() const { return macroAverage(&ConfusionMatrix::recall); }

	/**
	 * @brief Unweighted mean of per-class F1 over all classes.
	 */
	double macroF1() const { return macroAverage(&ConfusionMatrix::f1); }

private:
	std::size_t classCount;
	std::size_t totalCount = 0;
	std::vector<std::size_t> counts;

	static double ratio(std::size_t numerator, std::size_t denominator) {
		return denominator == 0 ? 0.0
			: static_cast<double>(numerator) / static_cast<double>(denominator);
	}

	double macroAverage(double (ConfusionMatrix::* score)(std::size_t) const) const {
		double sum = 0.0;
		for (std::size_t c = 0; c < classCount; c++) {
			sum += (this->*score)(c);
		}
		return sum / static_cast<double>(classCount);
	}
};
