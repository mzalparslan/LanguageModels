#pragma once

#include "Tensor.h"
#include "Validation.h"
#include <algorithm>
#include <cmath>

/**
 * @brief Matrix multiplication: C = A * B.
 *
 * A: [M, K], B: [K, N] => C: [M, N].
 * column count of A must equal row count of B.
 *
 * @param A Left operand.
 * @param B Right operand.
 * @param C Result; reused in place when it already has shape [M, N],
 * otherwise reallocated.
 * @throws InvalidSizeError If A or B is not a non-empty matrix, or A's
 * columns differ from B's rows.
 */
template <typename T>
void MatMul2D(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C) {
	validation::requireMatrix(A, "MatMul2D left operand");
	validation::requireMatrix(B, "MatMul2D right operand");

	std::size_t M = A.shape[0];
	std::size_t K = A.shape[1];
	std::size_t N = B.shape[1];

	if (K != B.shape[0]) {
		throw InvalidSizeError(
			"Column count of A should be equal to row count of B!");
	}

	// Reuse C if possible to save memory allocations. It must be cleared
	// because loops below accumulate (+=) into it.
	if (C.shape.size() == 2 &&
		C.shape[0] == M &&
		C.shape[1] == N) {
		std::fill(C.data.begin(), C.data.end(), T(0));
	}
	// Create new if size mismatch.
	else {
		C = Tensor<T>({ M, N }, T(0));
	}

	// i-k-j loop order: innermost loop walks rows of B and C
	// contiguously (row-major), which is cache friendly.
	for (std::size_t i = 0; i < M; i++) {
		for (std::size_t k = 0; k < K; k++) {
			T val = A.data[i * K + k];
			// Sparse optimization: a zero entry contributes nothing, so skip
			// its whole row of work (helps after ReLU, which zeroes many values).
			if (0 == val) {
				continue;
			}

			for (std::size_t j = 0; j < N; j++) {
				C.data[i * N + j] += val * B.data[k * N + j];
			}
		}
	}
}

/**
 * @brief Matrix transpose: At = A^T.
 *
 * A: [rows, cols] => At: [cols, rows].
 *
 * @param A Input matrix.
 * @param At Result; reused in place when it already has shape [cols, rows],
 * otherwise reallocated.
 * @throws InvalidSizeError If A is not a non-empty matrix.
 */
template <typename T>
void transpose2D(const Tensor<T>& A, Tensor<T>& At) {
	validation::requireMatrix(A, "transpose2D input");

	// Note: rowSize is A's row length (its column count), and colSize is
	// A's column length (its row count).
	std::size_t rowSize = A.shape[1];
	std::size_t colSize = A.shape[0];

	// Create new if size mismatch.
	if (At.shape.size() != 2 ||
		At.shape[0] != rowSize ||
		At.shape[1] != colSize) {
		At = Tensor<T>({ rowSize, colSize }, T(0));
	}

	// Element (i, j) of A moves to element (j, i) of At. Every element of At
	// is overwritten, so no clearing is needed.
	for (std::size_t i = 0; i < colSize; i++) {
		for (std::size_t j = 0; j < rowSize; j++) {
			At.data[j * colSize + i] = A.data[i * rowSize + j];
		}
	}
}

/**
 * @brief Row-wise softmax, in place: every row of an [M, N] matrix is turned
 * into a probability distribution (non-negative, sums to 1).
 *
 * @param mat Matrix of scores; overwritten with probabilities.
 * @throws InvalidSizeError If mat is not a non-empty matrix.
 * @throws NaNError, NonFiniteError If a row contains NaN or overflows.
 * @throws DivisionByZeroError If a row's normalizer is zero.
 */
template <typename T>
void softmaxRow(Tensor<T>& mat) {
	validation::requireMatrix(mat, "softmaxRow input");

	std::size_t M = mat.shape[0];
	std::size_t N = mat.shape[1];

	for (std::size_t i = 0; i < M; i++) {
		// Subtract row maximum before exponentiating: softmax is
		// unchanged by a constant shift, but this keeps exp() from
		// overflowing on large scores.
		T maxVal = -1e9;
		for (std::size_t j = 0; j < N; j++) {
			maxVal = std::max(maxVal, mat.data[i * N + j]);
		}

		// Exponentiate and accumulate normalizer.
		T sum = T(0);
		for (std::size_t j = 0; j < N; j++) {
			mat.data[i * N + j] = std::exp(mat.data[i * N + j] - maxVal);
			sum += mat.data[i * N + j];
		}

		// A NaN or overflowing score makes the normalizer NaN/Inf; a zero
		// normalizer would divide by zero below. Fail here rather than
		// spreading NaN through every later value.
		validation::requireFinite(sum, "Softmax normalizer");
		validation::requireNonZeroDenominator(sum, "softmax normalizer");

		// Normalize so row sums to 1.
		for (std::size_t j = 0; j < N; j++) {
			mat.data[i * N + j] /= sum;
		}
	}
}
