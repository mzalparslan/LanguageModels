#pragma once

// Host-side helpers shared by the .cu files of LanguageModels.Cuda.

#include "CudaRuntime.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

namespace cuda {

	namespace detail {

		/**
		 * @brief Throws CudaError if a CUDA runtime call did not succeed.
		 */
		inline void check(cudaError_t status, const char* what) {
			if (status != cudaSuccess) {
				throw CudaError(std::string(what) + " failed: " + cudaGetErrorString(status));
			}
		}

		// Threads per block for the one-thread-per-element kernels.
		constexpr unsigned blockSize = 256;

		inline unsigned blocksFor(std::size_t count) {
			return static_cast<unsigned>((count + blockSize - 1) / blockSize);
		}

	} // namespace detail

} // namespace cuda
