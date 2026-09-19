#pragma once

#include <stdexcept>

/**
 * @brief A computed value is not finite (Inf), e.g. a loss that overflowed or
 * a probability of exactly zero that reached log().
 *
 * NaNError is the more specific case. Both derive from std::domain_error.
 */
class NonFiniteError : public std::domain_error {
public:
	using std::domain_error::domain_error;
};

/**
 * @brief A computed value is NaN (not a number). NaN spreads through every
 * later calculation it touches, so it is reported the moment it is seen.
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
 * learning rate, a token id at or beyond the vocabulary size, a class label
 * out of range, and so on.
 */
class InvalidParameterError : public std::invalid_argument {
public:
	using std::invalid_argument::invalid_argument;
};

/**
 * @brief Data has the wrong size or shape for the operation: a matrix whose
 * width does not match a layer, two vectors that must be the same length, an
 * empty input, or a sequence longer than the model supports.
 */
class InvalidSizeError : public std::invalid_argument {
public:
	using std::invalid_argument::invalid_argument;
};

/**
 * @brief A model configuration size is zero or inconsistent: a zero width,
 * vocabulary or layer count, a model width not divisible by the number of
 * heads, more experts requested per token than exist, or a gradient whose
 * size differs from its weights.
 */
class InvalidParameterSizeError : public InvalidSizeError {
public:
	using InvalidSizeError::InvalidSizeError;
};
