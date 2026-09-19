#pragma once

#include "DecoderOnlyModel.h"
#include "BasicDecoderBlock.h"
#include "BasicGPTConfig.h"

/**
 * @brief GPT Components (Decoder-Only), plain single-head attention.
 *
 * Standard Decoder Block: Pre-Norm
 * x = x + Attn(Norm(x))
 * x = x + FF(Norm(x))
 *
 * A thin alias over DecoderOnlyModel -- see models/DecoderOnlyModel.h.
 */
using BasicGPT = DecoderOnlyModel<double, BasicDecoderBlock<double>, BasicGPTConfig>;
