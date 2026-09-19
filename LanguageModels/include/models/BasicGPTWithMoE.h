#pragma once

#include "DecoderOnlyModel.h"
#include "DecoderWithMoe.h"
#include "BasicGPTConfig.h"
#include "GPTWithUnigramConfig.h"

/**
 * @brief GPT Components (Decoder-Only): single-head attention, FeedForward
 * or Mixture-of-Experts per layer (see DecoderWithMoe).
 *
 * A thin alias over DecoderOnlyModel -- see models/DecoderOnlyModel.h.
 * Sized for TestBasicGPTWithMoE.cpp (d_model=16, ctx_len=20).
 */
using BasicGPTWithMoE = DecoderOnlyModel<double, DecoderWithMoe<double>, BasicGPTConfig>;

/**
 * @brief Same block type as BasicGPTWithMoE, sized instead for the
 * GPTWithUnigram stage (TestGPTWithUnigram.cpp), which trains at a
 * different d_model/ctx_len (32/32) and so needs its own RotaryEmbedding
 * config -- see config/GPTWithUnigramConfig.h.
 */
using GPTWithUnigram = DecoderOnlyModel<double, DecoderWithMoe<double>, GPTWithUnigramConfig>;
