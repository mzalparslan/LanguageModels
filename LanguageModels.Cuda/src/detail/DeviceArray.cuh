#pragma once

#include "CudaHost.cuh"

#include <cstddef>

namespace cuda {

	namespace detail {

		/**
		 * @brief An array in GPU memory that is freed when the object goes away. Grows
		 * (discarding its contents) when asked for more than it holds.
		 */
		template <typename E>
		class DeviceArray {
		public:
			DeviceArray() = default;
			~DeviceArray() { release(); }

			DeviceArray(const DeviceArray&) = delete;
			DeviceArray& operator=(const DeviceArray&) = delete;
			DeviceArray(DeviceArray&& other) noexcept : pointer(other.pointer), capacity(other.capacity) {
				other.pointer = nullptr;
				other.capacity = 0;
			}
			DeviceArray& operator=(DeviceArray&& other) noexcept {
				if (this != &other) {
					release();
					pointer = other.pointer;
					capacity = other.capacity;
					other.pointer = nullptr;
					other.capacity = 0;
				}
				return *this;
			}

			void allocate(std::size_t count) {
				release();
				check(cudaMalloc(&pointer, count * sizeof(E)), "cudaMalloc");
				capacity = count;
			}

			void ensure(std::size_t count) {
				if (count > capacity) {
					allocate(count);
				}
			}

			void zero(std::size_t count) {
				check(cudaMemset(pointer, 0, count * sizeof(E)), "cudaMemset");
			}

			E* get() const { return pointer; }

		private:
			E* pointer = nullptr;
			std::size_t capacity = 0;

			void release() {
				if (pointer != nullptr) {
					cudaFree(pointer);
					pointer = nullptr;
					capacity = 0;
				}
			}
		};

	} // namespace detail

} // namespace cuda
