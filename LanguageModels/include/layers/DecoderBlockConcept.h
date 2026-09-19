#pragma once

#include <concepts>
#include <cstddef>

#include "Tensor.h"
#include "RotaryEmbedding.h"

/**
 * @brief A decoder block usable as DecoderOnlyModel's BlockT: caches its own
 * activations for backprop, and exposes a forward pass (optionally rotating
 * Q/K with a RotaryEmbedding sized by RopeConfig), a backward pass, and
 * gradient zeroing/update. Satisfied by BasicDecoderBlock, DecoderWithMoe
 * and DecoderMultiHead.
 *
 * @tparam B Candidate block type.
 * @tparam T Floating-point mode.
 * @tparam RopeConfig Config type sizing RotaryEmbedding passed to forward().
 */
template <typename B, typename T, typename RopeConfig>
concept DecoderBlock = requires(B block, const Tensor<T>& x, Tensor<T>& out,
    const Tensor<T>& dout, Tensor<T>& dx,
    RotaryEmbedding<T, RopeConfig>* rope, T lr) {
    block.forward(x, out, rope);
    block.backward(dout, dx);
    block.zeroGrad();
    block.update(lr);
};

/**
 * @brief Optional capability: block tracks a Mixture-of-Experts
 * load-balancing auxiliary loss (DecoderWithMoe-style blocks), which
 * DecoderOnlyModel adds to its training loss when present.
 */
template <typename B, typename T>
concept HasAuxLoss = requires(const B& block) {
    { block.getAuxLoss() } -> std::convertible_to<T>;
};
