#pragma once

#include <cstddef>
#include <string>

#include "Exceptions.h"

/**
 * @brief Interface to the optional Metal (Apple GPU) code.
 *
 * This header is plain C++: it includes nothing from Apple's frameworks, so any
 * project can include it. Where the code lives depends on the platform:
 *  - on Apple platforms the functions are implemented in
 *    LanguageModels.Metal/src/MetalRuntime.mm (Objective-C++, linked with the
 *    Metal and Foundation frameworks) and use the system's default GPU;
 *  - everywhere else the functions below are inline stand-ins: isAvailable() is
 *    false and the others throw MetalError, so callers can fall back to the CPU.
 *
 * Always check isAvailable() before relying on the GPU. A Mac can still report
 * false: a virtual machine, such as a CI runner, may have no GPU to give.
 */
namespace metal {

	/**
	 * @brief What is known about the GPU in use (the system's default device).
	 */
	class DeviceInfo {
	public:
		std::string name;
		// The most memory the device advises using, in bytes (all of it on Apple Silicon,
		// where the CPU and the GPU share memory).
		std::size_t recommendedMemoryBytes = 0;
		// True when the CPU and the GPU share one memory (Apple Silicon).
		bool unifiedMemory = false;
	};

#if defined(__APPLE__)

	/**
	 * @brief Whether a Metal device is present. Never throws.
	 */
	bool isAvailable();

	/**
	 * @brief Describes the GPU in use.
	 *
	 * @throws MetalError If Metal is not available.
	 */
	DeviceInfo deviceInfo();

#else

	inline bool isAvailable() { return false; }

	inline DeviceInfo deviceInfo() {
		throw MetalError("metal::deviceInfo: Metal is only available on Apple platforms.");
	}

#endif

	/**
	 * @brief The GPU as one line of text, e.g. "Apple M1 (Virtual), 5461 MiB, unified memory",
	 * or a note saying Metal is unavailable. Never throws.
	 */
	inline std::string describeDevice() {
		if (!isAvailable()) {
			return "Metal is not available";
		}
		const DeviceInfo info = deviceInfo();
		return info.name + ", " + std::to_string(info.recommendedMemoryBytes / (1024 * 1024)) + " MiB"
			+ (info.unifiedMemory ? ", unified memory" : "");
	}

} // namespace metal
