// The GPU implementation of CudaRuntime.h. Compiled by nvcc (see nvcc-build.cmd);
// only built when the CUDA Toolkit is installed.

#include "CudaRuntime.h"
#include "detail/CudaHost.cuh"

#include <cuda_runtime.h>

#include <string>

namespace cuda {

	namespace {

		using detail::check;

		/**
		 * @brief Memory on the GPU, freed when the object goes out of scope, so
		 * an exception between allocation and use cannot leak it.
		 */
		template <typename T>
		class DeviceBuffer {
		public:
			explicit DeviceBuffer(std::size_t count) {
				check(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
			}
			~DeviceBuffer() { cudaFree(pointer); }

			DeviceBuffer(const DeviceBuffer&) = delete;
			DeviceBuffer& operator=(const DeviceBuffer&) = delete;

			T* get() const { return pointer; }

		private:
			T* pointer = nullptr;
		};

		// One thread per element.
		__global__ void saxpyKernel(std::size_t count, float a, const float* x, float* y) {
			const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			if (i < count) {
				y[i] = a * x[i] + y[i];
			}
		}

	} // namespace

	bool isAvailable() {
		int count = 0;
		if (cudaGetDeviceCount(&count) != cudaSuccess) {
			// Clear the error state, so it is not reported by a later, unrelated call.
			(void)cudaGetLastError();
			return false;
		}
		return count > 0;
	}

	DeviceInfo deviceInfo() {
		if (!isAvailable()) {
			throw CudaError("cuda::deviceInfo: No CUDA device is available.");
		}

		cudaDeviceProp properties;
		check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");

		DeviceInfo info;
		info.name = properties.name;
		info.totalMemoryBytes = properties.totalGlobalMem;
		info.computeMajor = properties.major;
		info.computeMinor = properties.minor;
		info.multiprocessorCount = properties.multiProcessorCount;
		return info;
	}

	void saxpy(float a, const std::vector<float>& x, std::vector<float>& y) {
		if (x.size() != y.size()) {
			throw InvalidSizeError("cuda::saxpy: x and y differ in length!");
		}
		if (x.empty()) {
			return;
		}
		if (!isAvailable()) {
			throw CudaError("cuda::saxpy: No CUDA device is available.");
		}

		const std::size_t count = x.size();
		DeviceBuffer<float> deviceX(count);
		DeviceBuffer<float> deviceY(count);
		check(cudaMemcpy(deviceX.get(), x.data(), count * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy (x to device)");
		check(cudaMemcpy(deviceY.get(), y.data(), count * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy (y to device)");

		const unsigned threadsPerBlock = 256;
		const unsigned blocks = static_cast<unsigned>((count + threadsPerBlock - 1) / threadsPerBlock);
		saxpyKernel<<<blocks, threadsPerBlock>>>(count, a, deviceX.get(), deviceY.get());
		check(cudaGetLastError(), "saxpy kernel launch");

		// cudaMemcpy waits for the kernel to finish before copying.
		check(cudaMemcpy(y.data(), deviceY.get(), count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy (y to host)");
	}

} // namespace cuda
