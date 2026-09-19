#pragma once

#include <cstddef>

/**
 * @brief MiniTransformer's own config, analogous to how real per-architecture
 * implementations (nanoGPT's GPTConfig, HuggingFace's BertConfig, ...) give
 * each model its own concrete config type rather than leaning on a shared
 * default.
 *
 * RotaryEmbedding sources its per-row dimension from Config::d_head
 * at compile time (see embeddings/RotaryEmbedding.h). ModelConfig's default
 * d_head is 64, but MiniTransformer's attention heads run at d_model=32, so
 * RotaryEmbedding<T> configured with ModelConfig would index Q/K rows past
 * their actual 32-wide stride -- an out-of-bounds write. This config matches
 * RoPE's dimension to MiniTransformer's own.
 */
class MiniTransformerConfig {
public:
	static constexpr std::size_t d_head = 32;
	static constexpr std::size_t maxSeqLen = 100;
};
