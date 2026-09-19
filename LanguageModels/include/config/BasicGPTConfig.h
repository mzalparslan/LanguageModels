#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @brief Config for BasicGPT / BasicGPTWithMoE, sized to match how
 * TestBasicGPT.cpp and TestBasicGPTWithMoE.cpp actually construct them
 * (d_model=16, ctx_len=20, single-head attention so d_head == d_model).
 *
 * See config/ModelConfig.h's header comment for why this project passes a
 * compile-time config type (nanoGPT/HuggingFace style) rather than sizing
 * RotaryEmbedding purely at runtime: RotaryEmbedding<T, Config>'s cos/sin
 * cache is precomputed once from Config::d_head/maxSeqLen, so it must match
 * whatever d_model/context length model is actually constructed with.
 */
class BasicGPTConfig {
public:
    static constexpr std::size_t d_head = 16;
    static constexpr std::size_t maxSeqLen = 20;
    // Seed for initial weights; same seed always gives same model.
    static constexpr std::uint32_t randomSeed = 42;
};
