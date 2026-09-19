#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "Exceptions.h"
#include "Tensor.h"

/**
 * @brief Argument and numerical checks shared by the whole library. Each
 * function throws the exception type that names the problem (see
 * Exceptions.h) and includes `what` in its message so the failing argument
 * is identifiable.
 */
namespace validation {

	/**
	 * @brief Requires a finite value.
	 *
	 * @throws NaNError If value is NaN.
	 * @throws NonFiniteError If value is +/- infinity.
	 */
	template <typename T>
	void requireFinite(T value, const char* what) {
		static_assert(std::is_floating_point_v<T>, "requireFinite needs a floating-point type");
		if (std::isnan(value)) {
			throw NaNError(std::string(what) + " is NaN!");
		}
		if (!std::isfinite(value)) {
			throw NonFiniteError(std::string(what) + " is infinite!");
		}
	}

	/**
	 * @brief Requires every element of a tensor to be finite.
	 *
	 * @throws NaNError, NonFiniteError See requireFinite().
	 */
	template <typename T>
	void requireAllFinite(const Tensor<T>& tensor, const char* what) {
		for (std::size_t i = 0; i < tensor.data.size(); i++) {
			requireFinite(tensor.data[i], what);
		}
	}

	/**
	 * @brief Requires a strictly positive, finite floating-point argument
	 * (a learning rate, an epsilon, ...).
	 *
	 * @throws NaNError, NonFiniteError If value is not finite.
	 * @throws InvalidParameterError If value <= 0.
	 */
	template <typename T>
	void requirePositiveFinite(T value, const char* what) {
		requireFinite(value, what);
		if (!(value > T(0))) {
			throw InvalidParameterError(std::string(what) + " must be greater than zero!");
		}
	}

	/**
	 * @brief Requires a non-negative, finite floating-point argument.
	 *
	 * @throws NaNError, NonFiniteError If value is not finite.
	 * @throws InvalidParameterError If value < 0.
	 */
	template <typename T>
	void requireNonNegativeFinite(T value, const char* what) {
		requireFinite(value, what);
		if (value < T(0)) {
			throw InvalidParameterError(std::string(what) + " must not be negative!");
		}
	}

	/**
	 * @brief Requires a denominator that is not zero.
	 *
	 * @throws DivisionByZeroError If denominator == 0.
	 */
	template <typename T>
	void requireNonZeroDenominator(T denominator, const char* what) {
		if (denominator == T(0)) {
			throw DivisionByZeroError(std::string("Division by zero: ") + what + " is zero!");
		}
	}

	/**
	 * @brief Requires a model configuration size (width, vocabulary, layer
	 * count, ...) to be greater than zero.
	 *
	 * @throws InvalidParameterSizeError If value == 0.
	 */
	inline void requirePositiveSize(std::size_t value, const char* what) {
		if (value == 0) {
			throw InvalidParameterSizeError(std::string(what) + " must be greater than zero!");
		}
	}

	/**
	 * @brief Requires a valid tensor shape: at least one dimension, none of
	 * them zero.
	 *
	 * @throws InvalidParameterSizeError If the shape is empty or has a zero dimension.
	 */
	inline void requireValidShape(const std::vector<std::size_t>& shape, const char* what) {
		if (shape.empty()) {
			throw InvalidParameterSizeError(std::string(what) + " must have at least one dimension!");
		}
		for (std::size_t dimension : shape) {
			if (dimension == 0) {
				throw InvalidParameterSizeError(std::string(what) + " must not have a zero dimension!");
			}
		}
	}

	/**
	 * @brief Requires that one configuration size divides another evenly.
	 *
	 * @throws InvalidParameterSizeError If divisor is 0 or total % divisor != 0.
	 */
	inline void requireDivisible(std::size_t total, std::size_t divisor,
		const char* totalName, const char* divisorName) {
		if (divisor == 0 || total % divisor != 0) {
			throw InvalidParameterSizeError(std::string(totalName)
				+ " must be divisible by " + divisorName + "!");
		}
	}

	/**
	 * @brief Requires an id or index to be below a limit (a token id below
	 * the vocabulary size, a class label below the class count, ...).
	 *
	 * @throws InvalidParameterError If index >= limit.
	 */
	inline void requireBelow(std::size_t index, std::size_t limit, const char* what) {
		if (index >= limit) {
			throw InvalidParameterError(std::string(what) + " " + std::to_string(index)
				+ " is out of range (must be below " + std::to_string(limit) + ")!");
		}
	}

	/**
	 * @brief Requires a data container to hold at least one element.
	 *
	 * @throws InvalidSizeError If size == 0.
	 */
	inline void requireNonEmpty(std::size_t size, const char* what) {
		if (size == 0) {
			throw InvalidSizeError(std::string(what) + " must not be empty!");
		}
	}

	/**
	 * @brief Requires a length not to exceed a maximum.
	 *
	 * @throws InvalidSizeError If size > maxSize.
	 */
	inline void requireAtMost(std::size_t size, std::size_t maxSize, const char* what) {
		if (size > maxSize) {
			throw InvalidSizeError(std::string(what) + " is " + std::to_string(size)
				+ " but at most " + std::to_string(maxSize) + " is supported!");
		}
	}

	/**
	 * @brief Requires two lengths to be equal.
	 *
	 * @throws InvalidSizeError If actual != expected.
	 */
	inline void requireSameSize(std::size_t actual, std::size_t expected, const char* what) {
		if (actual != expected) {
			throw InvalidSizeError(std::string(what) + " has size " + std::to_string(actual)
				+ " but " + std::to_string(expected) + " was expected!");
		}
	}

	/**
	 * @brief Requires a 2-D tensor (matrix) with at least one row and column.
	 *
	 * @throws InvalidSizeError If the tensor is not 2-D or is empty.
	 */
	template <typename T>
	void requireMatrix(const Tensor<T>& tensor, const char* what) {
		if (tensor.shape.size() != 2) {
			throw InvalidSizeError(std::string(what) + " must be a 2-D matrix!");
		}
		if (tensor.shape[0] == 0 || tensor.shape[1] == 0) {
			throw InvalidSizeError(std::string(what) + " must not be empty!");
		}
	}

	/**
	 * @brief Requires a matrix with an exact number of columns.
	 *
	 * @throws InvalidSizeError See requireMatrix(), or if the width differs.
	 */
	template <typename T>
	void requireColumns(const Tensor<T>& tensor, std::size_t columns, const char* what) {
		requireMatrix(tensor, what);
		requireSameSize(tensor.shape[1], columns, (std::string(what) + " column count").c_str());
	}

	/**
	 * @brief Requires a matrix with an exact number of rows and columns.
	 *
	 * @throws InvalidSizeError See requireColumns(), or if the row count differs.
	 */
	template <typename T>
	void requireShape(const Tensor<T>& tensor, std::size_t rows, std::size_t columns,
		const char* what) {
		requireColumns(tensor, columns, what);
		requireSameSize(tensor.shape[0], rows, (std::string(what) + " row count").c_str());
	}

} // namespace validation
