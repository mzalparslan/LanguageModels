#pragma once

#include <vector>
#include <limits>
#include <cstddef>
#include <stdexcept>

#include "Exceptions.h"

/**
 * @brief Flat 1D vector representing a multi-dimensional tensor.
 *
 * Exm: A 2x3 matrix [[1, 2, 3], [4, 5, 6]] is
 * stored as [1,2,3,4,5,6] with a Shape {2, 3}
 */
template <typename T>
class Tensor {
public:
	std::vector<T> data;
	std::vector<std::size_t> shape;

	Tensor() {}

	/**
	 * @throws InvalidSizeError In case size exceeds max limit.
	 */
	Tensor(const std::vector<std::size_t>& shape, T value = T(0))
		: shape(shape)
	{
		std::size_t rawSize = 1;

		for (auto shapeSize : shape) {
			// Check overflow.
			if ((shapeSize != 0) && 
				(std::numeric_limits<std::size_t>::max() / shapeSize) < rawSize) {
				throw InvalidSizeError("Overflow for tensor size!");
			}

			rawSize *= shapeSize;
		}

		data.assign(rawSize, value);
	}

	/**
	 * @brief Flat, rank-agnostic element access. Caller is responsible for
	 * computing correct flat offset.
	 */
	T& operator[](std::size_t i) { return data[i]; }
	const T& operator[](std::size_t i) const { return data[i]; }

	/**
	 * @brief Get total size of tensor.
	 */
	std::size_t size() const { return data.size(); }
};