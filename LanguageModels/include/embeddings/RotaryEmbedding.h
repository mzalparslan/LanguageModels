#pragma once

#include "Tensor.h"
#include "Validation.h"
#include "ModelConfig.h"

/**
 * @brief Rotary Embedding Class implementing RoPE mechanism.
 * Idea: Rotate tensor x any type of input with rotational position matrix.
 * Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
 * by Su et al., 2021.
 *
 * @remark RoPE encodes positional information by rotating input vectors
 * based on their position using precomputed sine and cosine values.
 * This allows model to capture relative positional relationships
 * more effectively than traditional absolute positional embeddings.
 */
template <typename T, typename RopeConfig = ModelConfig>
class RotaryEmbedding {
public:
    /**
	 * @brief Rotary Embedding Constructor initializes caches for cosine and sine values.
     */
    RotaryEmbedding() {
		// Touch both tables so they are built now rather than on first apply().
		// They are static: shared by every RotaryEmbedding of same
		// <T, RopeConfig>, so constructing more instances costs nothing.
        getCosCache();
        getSinCache();
    }

    /**
	 * @brief Rotate input tensor x in-place using precomputed rotary embeddings.
     *
     * @param x Tensor to be rotated [seq, dHead], where dHead must equal
     * RopeConfig::d_head: row stride is taken from config, not from x.
     * Position p of sequence uses angle p * theta_i.
     * @throws InvalidSizeError If x is not a [seq, RopeConfig::d_head] matrix, or seq
     * exceeds RopeConfig::maxSeqLen.
     */
    void apply(Tensor<T>& x) {
        rotate(x, T(1));
    }

    /**
     * @brief Undo apply(): rotates x in-place by opposite angles.
     *
     * A rotation is orthogonal, so its inverse is its transpose. This is what
     * backpropagation needs: if y = apply(x), then gradient with respect
     * to x is applyInverse(gradient with respect to y).
     *
     * @param x Tensor to be rotated back [seq, dHead]; same requirements as apply().
     * @throws InvalidSizeError See apply().
     */
    void applyInverse(Tensor<T>& x) {
        rotate(x, T(-1));
    }

private:
	// Positions are encoded by rotating pairs of dimensions, so head
	// width must be positive and even; a zero max length has no positions.
	static_assert(RopeConfig::d_head > 0 && RopeConfig::d_head % 2 == 0,
		"RotaryEmbedding needs a positive, even d_head!");
	static_assert(RopeConfig::maxSeqLen > 0,
		"RotaryEmbedding needs a positive maxSeqLen!");

	// Extracted dimension size for rotary embedding from Config.
	static const std::size_t dimension = RopeConfig::d_head;
	// Extracted maximum sequence length from Config.
    static const std::size_t maxLength  = RopeConfig::maxSeqLen;
	// As there are paired sin/cos values, cache dimension is half of total dimension.
    static const std::size_t cacheDim = dimension / 2;

	// Cosine table [maxLength, cacheDim]: entry (pos, i) = cos(pos * theta_i).
	// getCosCache() builds it together with sine table.
    static Tensor<T>& getCosCache() {
		// Meyers' singleton: a function-local static built on first use, shared
		// by all instances of this <T, RopeConfig>.
        static Tensor<T> cache;
        if (cache.data.empty()) {
            initCaches(cache, getSinCache(true));
        }

        return cache;
    }

	// Sine table [maxLength, cacheDim]: entry (pos, i) = sin(pos * theta_i).
	// Filled by initCaches() when cosine table is first built.
    static Tensor<T>& getSinCache(bool init = false) {
        static Tensor<T> cache;
        if (!init && cache.data.empty()) {
            // If accessed directly, ensure initialized. 
            // Note: usually getCosCache handles both.
            getCosCache();
        }
        return cache;
    }

	/// Fills both tables for every (position, frequency-pair) combination.
    static void initCaches(Tensor<T>& cosCache, Tensor<T>& sinCache) {

        cosCache = Tensor<T>({ maxLength, cacheDim }, 0);
        sinCache = Tensor<T>({ maxLength, cacheDim }, 0);

        for (std::size_t pos = 0; pos < maxLength; ++pos) {
            for (std::size_t i = 0; i < cacheDim; i++) {
                // theta_i = 10000^(-2i/d): pair i rotates at a geometric frequency,
                // so early pairs turn fast (local order) and late pairs slowly
                // (long range), like sinusoidal encoding.
                double theta = std::pow(10000.0, -2.0 * i / dimension);
                double val = pos * theta;
                // Worked out in double whatever T is, then rounded once to T.
                cosCache.data[pos * cacheDim + i] = static_cast<T>(std::cos(val));
                sinCache.data[pos * cacheDim + i] = static_cast<T>(std::sin(val));
            }
        }
    }

    /**
     * @brief Shared implementation of apply() and applyInverse().
     *
     * @param x Tensor to be rotated in-place.
     * @param direction +1 rotates forward by each position's angle, -1 by the
     * opposite angle (sin(-a) = -sin(a), cos(-a) = cos(a)).
     */
    void rotate(Tensor<T>& x, T direction) {
		validation::requireColumns(x, dimension, "RotaryEmbedding input");
		// Validate input dimensions
        std::size_t seq = x.shape[0];
        if (seq > maxLength) {
			throw InvalidSizeError("Input sequence length exceeds maximum length for Rotary Embedding.");
        }

        // Access static caches
        Tensor<T>& cosCache = getCosCache();
        Tensor<T>& sinCache = getSinCache();

		// Apply rotation: each consecutive pair (2i, 2i + 1) of a row is treated
		// as a 2D vector and rotated by an angle proportional to its position.
		// Rotating both Q and K this way makes their dot product depend only on
		// position difference, i.e. attention sees relative position.
        for (std::size_t p = 0; p < seq; p++) {
            for (std::size_t i = 0; i < cacheDim; i++) {
                std::size_t idx0 = p * dimension + 2 * i;
                std::size_t idx1 = p * dimension + 2 * i + 1;

                T v0 = x.data[idx0];
                T v1 = x.data[idx1];

                T cosX = cosCache.data[p * cacheDim + i];
                T sinX = direction * sinCache.data[p * cacheDim + i];

                // Standard 2D rotation by angle stored in tables.
                x.data[idx0] = v0 * cosX - v1 * sinX;
                x.data[idx1] = v0 * sinX + v1 * cosX;
            }
        }
    }
};