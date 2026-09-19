#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @brief Config for GPTWithUnigram stage (TestGPTWithUnigram.cpp),
 * which trains a DecoderWithMoe-based model at d_model=32, ctx_len=32 --
 * a different size than BasicGPTConfig.h's model, so it needs its own
 * RotaryEmbedding sizing (see BasicGPTConfig.h for why a shared runtime
 * size can't be used here).
 */
class GPTWithUnigramConfig {
public:
    static constexpr std::size_t d_head = 32;
    static constexpr std::size_t maxSeqLen = 32;
    // Seed for initial weights; same seed always gives same model.
    static constexpr std::uint32_t randomSeed = 42;
};
