//
// TestSupport.h
//
// Helpers shared by unit tests: tensor construction/comparison,
// finite-difference gradient checking and console silencing.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "Tensor.h"

namespace testsupport {

	/**
	 * @brief Reads a data file from repository's resources/ folder.
	 *
	 * tests run from different directories (the solution's bin folder under
	 * Visual Studio, repository root or build/ folder under make), so this
	 * searches working directory and each parent for resources/<name>, or
	 * for <name> next to executable.
	 *
	 * @return file's bytes, or an empty string if it was not found.
	 */
	inline std::string readResource(const std::string& name) {
		namespace fs = std::filesystem;

		fs::path directory = fs::current_path();
		for (int level = 0; level < 8; level++) {
			for (const fs::path& candidate : { directory / "resources" / name, directory / name }) {
				if (fs::is_regular_file(candidate)) {
					std::ifstream file(candidate, std::ios::binary);
					std::ostringstream contents;
					contents << file.rdbuf();
					return contents.str();
				}
			}
			if (!directory.has_parent_path() || directory.parent_path() == directory) {
				break;
			}
			directory = directory.parent_path();
		}
		return std::string();
	}

	/**
	 * @brief Builds a [rows, cols] tensor from a row-major initializer list.
	 */
	inline Tensor<double> makeMatrix(std::size_t rows, std::size_t cols,
		const std::vector<double>& values) {
		Tensor<double> tensor({ rows, cols });
		EXPECT_EQ(values.size(), rows * cols) << "makeMatrix: wrong number of values";
		for (std::size_t i = 0; i < tensor.size() && i < values.size(); i++) {
			tensor[i] = values[i];
		}
		return tensor;
	}

	/**
	 * @brief Builds a [rows, cols] tensor of small, deterministic, non-trivial
	 * values (a sine pattern), so tests do not need a random engine.
	 * @param phase Shifts pattern so different inputs can be produced.
	 */
	inline Tensor<double> patternMatrix(std::size_t rows, std::size_t cols,
		double phase = 0.0) {
		Tensor<double> tensor({ rows, cols });
		for (std::size_t i = 0; i < tensor.size(); i++) {
			tensor[i] = 0.5 * std::sin(0.7 * static_cast<double>(i) + phase)
				+ 0.1 * static_cast<double>(i % 3);
		}
		return tensor;
	}

	/**
	 * @brief Compares two tensors' shapes exactly and values within tolerance.
	 */
	inline ::testing::AssertionResult tensorsNear(const Tensor<double>& actual,
		const Tensor<double>& expected, double tolerance) {
		if (actual.shape != expected.shape) {
			return ::testing::AssertionFailure() << "shape differs";
		}
		for (std::size_t i = 0; i < actual.size(); i++) {
			if (!(std::fabs(actual[i] - expected[i]) <= tolerance)) {
				return ::testing::AssertionFailure() << "element " << i << ": "
					<< actual[i] << " vs " << expected[i];
			}
		}
		return ::testing::AssertionSuccess();
	}

	/**
	 * @brief True when both tensors have same shape and identical values.
	 */
	inline bool tensorsEqual(const Tensor<double>& left, const Tensor<double>& right) {
		return left.shape == right.shape && left.data == right.data;
	}

	/**
	 * @brief Sum of all elements weighted by `weights`: a scalar "loss" whose
	 * gradient with respect to `tensor` is exactly `weights`. Used to turn a
	 * layer's tensor output into a scalar for gradient checking.
	 */
	inline double weightedSum(const Tensor<double>& tensor, const Tensor<double>& weights) {
		double total = 0.0;
		for (std::size_t i = 0; i < tensor.size(); i++) {
			total += tensor[i] * weights[i];
		}
		return total;
	}

	/**
	 * @brief Central finite-difference derivative of `lossFunction` with
	 * respect to element `index` of `variable`, which is restored afterwards.
	 *
	 * @param variable Any tensor `lossFunction` reads (an input or a weight).
	 * @param lossFunction Recomputes scalar loss from current values.
	 */
	inline double numericGradient(Tensor<double>& variable, std::size_t index,
		const std::function<double()>& lossFunction, double step = 1e-6) {
		double original = variable[index];

		variable[index] = original + step;
		double plusLoss = lossFunction();
		variable[index] = original - step;
		double minusLoss = lossFunction();
		variable[index] = original;

		return (plusLoss - minusLoss) / (2.0 * step);
	}

	/**
	 * @brief Checks an analytic gradient against finite differences for every
	 * element of `variable`.
	 *
	 * @param analytic Gradient computed by code under test (same shape).
	 * @param tolerance Allowed |analytic - numeric|, relative to larger
	 * magnitude (floored at 1) so tiny gradients are compared absolutely.
	 */
	inline ::testing::AssertionResult gradientMatches(Tensor<double>& variable,
		const Tensor<double>& analytic,
		const std::function<double()>& lossFunction, double tolerance = 1e-5) {
		if (analytic.size() != variable.size()) {
			return ::testing::AssertionFailure() << "gradient has a different size";
		}
		for (std::size_t i = 0; i < variable.size(); i++) {
			double numeric = numericGradient(variable, i, lossFunction);
			double scale = std::max(1.0, std::max(std::fabs(numeric), std::fabs(analytic[i])));
			if (!(std::fabs(numeric - analytic[i]) <= tolerance * scale)) {
				return ::testing::AssertionFailure() << "element " << i
					<< ": analytic " << analytic[i] << " vs numeric " << numeric;
			}
		}
		return ::testing::AssertionSuccess();
	}

	/**
	 * @brief Swallows everything written to std::cout while it is alive.
	 * Some library routines (tokenizer training, vocabulary loading) print
	 * progress; this keeps test output readable.
	 */
	class SilenceStdout {
	public:
		SilenceStdout() : previous(std::cout.rdbuf(sink.rdbuf())) {}
		~SilenceStdout() { std::cout.rdbuf(previous); }
		SilenceStdout(const SilenceStdout&) = delete;
		SilenceStdout& operator=(const SilenceStdout&) = delete;

	private:
		std::ostringstream sink;
		std::streambuf* previous;
	};

	/**
	 * @brief Same as SilenceStdout, for std::cerr.
	 */
	class SilenceStderr {
	public:
		SilenceStderr() : previous(std::cerr.rdbuf(sink.rdbuf())) {}
		~SilenceStderr() { std::cerr.rdbuf(previous); }
		SilenceStderr(const SilenceStderr&) = delete;
		SilenceStderr& operator=(const SilenceStderr&) = delete;

	private:
		std::ostringstream sink;
		std::streambuf* previous;
	};

} // namespace testsupport
