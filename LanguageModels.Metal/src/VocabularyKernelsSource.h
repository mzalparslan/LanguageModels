#pragma once

// The Metal kernels of metal::VocabularyHead, in Metal Shading Language, as a string
// that is compiled when the first head is created.
//
// Why a string and not a .metal file built into a .metallib: compiling shaders ahead of
// time needs Apple's offline Metal compiler, which recent Xcode versions ship as an
// optional download. The runtime compiler is part of the operating system, so the build
// needs nothing beyond a C++ compiler and the Metal framework.
//
// These are the kernels of the CUDA version (LanguageModels.Cuda/src/detail/
// VocabularyKernels.cuh), line for line where Metal allows, in float:
//  - output scores: one thread per (position, word), summing over the width in order;
//  - softmax + cross-entropy: one threadgroup per position, a parallel max and sum;
//  - projection gradient: one thread per (width index, word), summing over positions in order;
//  - decoder-output gradient: one threadgroup per (position, width index), a parallel sum over words;
//  - embedding gradient: a single threadgroup, positions added in order (so a word that
//    occurs twice gets the same result every time, and no atomic additions are needed);
//  - optimizer: one thread per weight.
//
// Sizes and counts are 32-bit (uint); the host checks they fit. Every kernel is dispatched
// with threadgroups of 256 threads.

namespace metal {

	namespace detail {

		inline const char* vocabularyKernelsSource() {
			return R"MSL(
#include <metal_stdlib>
using namespace metal;

// Bits a kernel sets in the error flag when it meets a value that is not finite.
constant int flagNaN = 1;
constant int flagInfinite = 2;

inline float clipToUnit(float gradient) {
    if (gradient > 1.0f) {
        gradient = 1.0f;
    }
    if (gradient < -1.0f) {
        gradient = -1.0f;
    }
    return gradient;
}

inline void flagIfNotFinite(float value, device atomic_int* flag) {
    if (isnan(value)) {
        atomic_fetch_or_explicit(flag, flagNaN, memory_order_relaxed);
    }
    else if (isinf(value)) {
        atomic_fetch_or_explicit(flag, flagInfinite, memory_order_relaxed);
    }
}

// Reductions over one threadgroup. Every thread of the group must call them; shared
// holds one value per simdgroup (at most 32). Each simdgroup reduces its own values,
// then every thread combines the per-simdgroup results in the same order.
inline float blockMax(float value, threadgroup float* shared, uint lane, uint simdIndex, uint simdCount) {
    value = simd_max(value);
    if (lane == 0) {
        shared[simdIndex] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = shared[0];
    for (uint i = 1; i < simdCount; i++) {
        result = max(result, shared[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline float blockSum(float value, threadgroup float* shared, uint lane, uint simdIndex, uint simdCount) {
    value = simd_sum(value);
    if (lane == 0) {
        shared[simdIndex] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = shared[0];
    for (uint i = 1; i < simdCount; i++) {
        result += shared[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// out[r] = table[ids[r]], one row per id.
kernel void gatherRowsKernel(device const float* table [[buffer(0)]],
                             device const int* ids [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& width [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    if (i < rows * width) {
        out[i] = table[uint(ids[i / width]) * width + i % width];
    }
}

// Adds rowGradients[r] into grad[ids[r]] for r in order. One threadgroup; each thread
// owns some columns, so every element is added in position order.
kernel void scatterAddRowsKernel(device float* grad [[buffer(0)]],
                                 device const int* ids [[buffer(1)]],
                                 device const float* rowGradients [[buffer(2)]],
                                 constant uint& rows [[buffer(3)]],
                                 constant uint& width [[buffer(4)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint threads [[threads_per_threadgroup]]) {
    for (uint column = tid; column < width; column += threads) {
        for (uint r = 0; r < rows; r++) {
            grad[uint(ids[r]) * width + column] += rowGradients[r * width + column];
        }
    }
}

// logits[r][v] = sum over k of x[r][k] * W[k][v], plus bias[v]. Grid: x over words, y over positions.
kernel void projectKernel(device const float* x [[buffer(0)]],
                          device const float* weights [[buffer(1)]],
                          device const float* bias [[buffer(2)]],
                          device float* logits [[buffer(3)]],
                          constant uint& width [[buffer(4)]],
                          constant uint& vocab [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const uint v = gid.x;
    const uint r = gid.y;
    if (v < vocab) {
        float sum = 0.0f;
        for (uint k = 0; k < width; k++) {
            sum += x[r * width + k] * weights[k * vocab + v];
        }
        logits[r * vocab + v] = sum + bias[v];
    }
}

// One threadgroup per position. Turns the row of logits into (probabilities - one-hot label),
// which is the gradient of the cross-entropy loss with respect to the logits, and
// records -log(probability of the label).
kernel void softmaxCrossEntropyKernel(device float* logits [[buffer(0)]],
                                      device const int* labels [[buffer(1)]],
                                      device float* rowLoss [[buffer(2)]],
                                      constant uint& vocab [[buffer(3)]],
                                      uint group [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint threads [[threads_per_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint simdIndex [[simdgroup_index_in_threadgroup]],
                                      uint simdCount [[simdgroups_per_threadgroup]]) {
    threadgroup float shared[32];
    device float* row = logits + group * vocab;

    float maxValue = -1e30f;
    for (uint v = tid; v < vocab; v += threads) {
        maxValue = max(maxValue, row[v]);
    }
    maxValue = blockMax(maxValue, shared, lane, simdIndex, simdCount);

    float sum = 0.0f;
    for (uint v = tid; v < vocab; v += threads) {
        const float e = exp(row[v] - maxValue);
        row[v] = e;
        sum += e;
    }
    sum = blockSum(sum, shared, lane, simdIndex, simdCount);

    for (uint v = tid; v < vocab; v += threads) {
        row[v] /= sum;
    }
    // Thread 0 reads the label's entry, which another thread may have written.
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        const int label = labels[group];
        const float probability = row[label];
        rowLoss[group] = -log(probability);
        row[label] = probability - 1.0f;
    }
}

// gradWeights[k][v] += sum over positions r of x[r][k] * dlogits[r][v];
// gradBias[v] += sum over r of dlogits[r][v]. Grid: x over words, y over width.
kernel void projectionGradientKernel(device const float* x [[buffer(0)]],
                                     device const float* dlogits [[buffer(1)]],
                                     device float* gradWeights [[buffer(2)]],
                                     device float* gradBias [[buffer(3)]],
                                     constant uint& rows [[buffer(4)]],
                                     constant uint& width [[buffer(5)]],
                                     constant uint& vocab [[buffer(6)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint v = gid.x;
    const uint k = gid.y;
    if (v < vocab) {
        float sum = 0.0f;
        for (uint r = 0; r < rows; r++) {
            sum += x[r * width + k] * dlogits[r * vocab + v];
        }
        gradWeights[k * vocab + v] += sum;

        if (k == 0) {
            float biasSum = 0.0f;
            for (uint r = 0; r < rows; r++) {
                biasSum += dlogits[r * vocab + v];
            }
            gradBias[v] += biasSum;
        }
    }
}

// dx[r][k] = sum over words v of dlogits[r][v] * W[k][v]. One threadgroup per (r, k).
kernel void inputGradientKernel(device const float* dlogits [[buffer(0)]],
                                device const float* weights [[buffer(1)]],
                                device float* dx [[buffer(2)]],
                                constant uint& width [[buffer(3)]],
                                constant uint& vocab [[buffer(4)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint threads [[threads_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simdIndex [[simdgroup_index_in_threadgroup]],
                                uint simdCount [[simdgroups_per_threadgroup]]) {
    threadgroup float shared[32];
    const uint r = group / width;
    const uint k = group % width;

    float sum = 0.0f;
    for (uint v = tid; v < vocab; v += threads) {
        sum += dlogits[r * vocab + v] * weights[k * vocab + v];
    }
    sum = blockSum(sum, shared, lane, simdIndex, simdCount);
    if (tid == 0) {
        dx[group] = sum;
    }
}

// value -= lr * clip(gradient). Reports values that are not finite through the flag.
kernel void sgdKernel(device float* value [[buffer(0)]],
                      device const float* gradient [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant float& learningRate [[buffer(3)]],
                      device atomic_int* flag [[buffer(4)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < count) {
        const float g = gradient[i];
        flagIfNotFinite(g, flag);
        const float updated = value[i] - learningRate * clipToUnit(g);
        flagIfNotFinite(updated, flag);
        value[i] = updated;
    }
}

// One Adam step, with the same arithmetic as Parameter::updateWAdamRange().
kernel void adamKernel(device float* value [[buffer(0)]],
                       device const float* gradient [[buffer(1)]],
                       device float* first [[buffer(2)]],
                       device float* second [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant float& learningRate [[buffer(5)]],
                       constant float& beta1 [[buffer(6)]],
                       constant float& beta2 [[buffer(7)]],
                       constant float& epsilon [[buffer(8)]],
                       constant float& biasCorrection1 [[buffer(9)]],
                       constant float& biasCorrection2 [[buffer(10)]],
                       device atomic_int* flag [[buffer(11)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < count) {
        float g = gradient[i];
        flagIfNotFinite(g, flag);
        g = clipToUnit(g);

        first[i] = beta1 * first[i] + (1.0f - beta1) * g;
        second[i] = beta2 * second[i] + (1.0f - beta2) * g * g;

        const float firstHat = first[i] / biasCorrection1;
        const float secondHat = second[i] / biasCorrection2;
        const float updated = value[i] - learningRate * firstHat / (sqrt(secondHat) + epsilon);
        flagIfNotFinite(updated, flag);
        value[i] = updated;
    }
}
)MSL";
		}

	} // namespace detail

} // namespace metal
