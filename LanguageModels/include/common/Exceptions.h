#pragma once

#include <stdexcept>

/**
 * @brief A computed value is not finite (Inf), e.g. a loss that overflowed or
 * a probability of exactly zero that reached log().
 *
 * NaNError is more specific case. Both derive from std::domain_error.
 */
class NonFiniteError : public std::domain_error {
public:
	using std::domain_error::domain_error;
};

/**
 * @brief A computed value is NaN (not a number). NaN spreads through every
 * later calculation it touches, so it is reported moment it is seen.
 */
class NaNError : public NonFiniteError {
public:
	using NonFiniteError::NonFiniteError;
};

/**
 * @brief A denominator was exactly zero (e.g. a softmax sum, a total weight, or
 * a count of predictions).
 */
class DivisionByZeroError : public std::domain_error {
public:
	using std::domain_error::domain_error;
};

/**
 * @brief An argument's value is outside its valid range: a non-positive
 * learning rate, a token id at or beyond vocabulary size, a class label
 * out of range, and so on.
 */
class InvalidParameterError : public std::invalid_argument {
public:
	using std::invalid_argument::invalid_argument;
};

/**
 * @brief Data has wrong size or shape for operation: a matrix whose
 * width does not match a layer, two vectors that must be same length, an
 * empty input, or a sequence longer than model supports.
 */
class InvalidSizeError : public std::invalid_argument {
public:
	using std::invalid_argument::invalid_argument;
};

/**
 * @brief A model configuration size is zero or inconsistent: a zero width,
 * vocabulary or layer count, a model width not divisible by number of
 * heads, more experts requested per token than exist, or a gradient whose
 * size differs from its weights.
 */
class InvalidParameterSizeError : public InvalidSizeError {
public:
	using InvalidSizeError::InvalidSizeError;
};

/**
 * @brief A data file could not be opened or read, or holds no usable data.
 * Derives from std::runtime_error: unlike argument errors above, the
 * arguments were fine and outside world (the file) was not.
 */
class DataLoadError : public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};

/**
 * @brief A pipeline was asked to do something out of order, such as evaluating
 * a model that has not been trained yet.
 */
class PipelineStateError : public std::logic_error {
public:
	using std::logic_error::logic_error;
};

/**
 * @brief A CUDA call failed, or CUDA was asked for but is not available (no
 * NVIDIA device, or the library was built without the CUDA Toolkit).
 */
class CudaError : public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};
