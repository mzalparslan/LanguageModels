#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Exceptions.h"

/**
 * @brief Interface to the optional CUDA (NVIDIA GPU) code.
 *
 * This header is plain C++: it includes nothing from the CUDA Toolkit, so any
 * project can include it and be compiled by MSVC or GCC alone. The
 * implementation lives in the LanguageModels.Cuda project, which links into
 * every build:
 *  - built with the CUDA Toolkit (CudaRuntime.cu): the functions run on the GPU;
 *  - built without it (CudaUnavailable.cpp): isAvailable() is false and the
 *    other functions throw CudaError, so callers can fall back to the CPU.
 *
 * Always check isAvailable() before relying on the GPU.
 */
namespace cuda {

	/**
	 * @brief What is known about the GPU in use (device 0).
	 */
	class DeviceInfo {
	public:
		std::string name;
		// Total memory of the device in bytes.
		std::size_t totalMemoryBytes = 0;
		// Compute capability, e.g. 8.9 for an RTX 4060 Ti (Ada).
		int computeMajor = 0;
		int computeMinor = 0;
		// Number of streaming multiprocessors.
		int multiprocessorCount = 0;
	};

	/**
	 * @brief Whether this build can use CUDA and a CUDA device is present.
	 * Never throws.
	 */
	bool isAvailable();

	/**
	 * @brief Describes the GPU in use.
	 *
	 * @throws CudaError If CUDA is not available.
	 */
	DeviceInfo deviceInfo();

	/**
	 * @brief The GPU as one line of text, e.g.
	 * "NVIDIA GeForce RTX 4060 Ti, 8188 MiB, compute capability 8.9, 34 multiprocessors",
	 * or a note saying CUDA is unavailable. Never throws.
	 */
	inline std::string describeDevice() {
		if (!isAvailable()) {
			return "CUDA is not available";
		}
		const DeviceInfo info = deviceInfo();
		return info.name + ", " + std::to_string(info.totalMemoryBytes / (1024 * 1024)) + " MiB, compute capability "
			+ std::to_string(info.computeMajor) + "." + std::to_string(info.computeMinor) + ", "
			+ std::to_string(info.multiprocessorCount) + " multiprocessors";
	}

	/**
	 * @brief y = a * x + y on the GPU, element by element.
	 *
	 * The first kernel of the project. It exercises the whole toolchain (nvcc,
	 * linking, the runtime, a real launch and the copies to and from the
	 * device) before the real work is built on it. The multiply and add are
	 * fused on the GPU, so for arbitrary values the result can differ from the
	 * CPU's by one rounding.
	 *
	 * @param a Scale factor.
	 * @param x Input, same length as y.
	 * @param y Input and result.
	 * @throws InvalidSizeError If x and y differ in length.
	 * @throws CudaError If CUDA is not available or a CUDA call fails.
	 */
	void saxpy(float a, const std::vector<float>& x, std::vector<float>& y);

} // namespace cuda
