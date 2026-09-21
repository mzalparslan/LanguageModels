#pragma once

// Small functions that run on the GPU and are used by more than one kernel.

#include <cuda_runtime.h>

namespace cuda {

	namespace detail {

		// Bits a kernel sets in the error flag when it meets a value that is not finite.
		constexpr int flagNaN = 1;
		constexpr int flagInfinite = 2;

		template <typename T>
		__device__ inline T clipToUnit(T gradient) {
			if (gradient > T(1)) {
				gradient = T(1);
			}
			if (gradient < T(-1)) {
				gradient = T(-1);
			}
			return gradient;
		}

		template <typename T>
		__device__ inline void flagIfNotFinite(T value, int* flag) {
			if (isnan(value)) {
				atomicOr(flag, flagNaN);
			}
			else if (isinf(value)) {
				atomicOr(flag, flagInfinite);
			}
		}

		// Parallel reductions over one block of blockSize threads; shared holds 32 values.
		template <typename T>
		__device__ T blockMax(T value, T* shared) {
			const int lane = threadIdx.x & 31;
			const int warp = threadIdx.x >> 5;
			for (int offset = 16; offset > 0; offset >>= 1) {
				value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
			}
			if (lane == 0) {
				shared[warp] = value;
			}
			__syncthreads();
			value = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : T(-1e30);
			if (warp == 0) {
				for (int offset = 16; offset > 0; offset >>= 1) {
					value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
				}
			}
			if (threadIdx.x == 0) {
				shared[0] = value;
			}
			__syncthreads();
			const T result = shared[0];
			__syncthreads();
			return result;
		}

		template <typename T>
		__device__ T blockSum(T value, T* shared) {
			const int lane = threadIdx.x & 31;
			const int warp = threadIdx.x >> 5;
			for (int offset = 16; offset > 0; offset >>= 1) {
				value += __shfl_down_sync(0xffffffffu, value, offset);
			}
			if (lane == 0) {
				shared[warp] = value;
			}
			__syncthreads();
			value = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : T(0);
			if (warp == 0) {
				for (int offset = 16; offset > 0; offset >>= 1) {
					value += __shfl_down_sync(0xffffffffu, value, offset);
				}
			}
			if (threadIdx.x == 0) {
				shared[0] = value;
			}
			__syncthreads();
			const T result = shared[0];
			__syncthreads();
			return result;
		}

	} // namespace detail

} // namespace cuda
