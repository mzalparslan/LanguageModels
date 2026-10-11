// The Apple implementation of MetalRuntime.h. Objective-C++: compiled with -fobjc-arc and
// linked with the Metal and Foundation frameworks (see the Makefile). Off Apple platforms
// this file is not built, and the inline stand-ins in the header are used instead.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalRuntime.h"

#include <string>

namespace metal {

	bool isAvailable() {
		bool present = false;
		@autoreleasepool {
			id<MTLDevice> device = MTLCreateSystemDefaultDevice();
			present = (device != nil);
		}
		return present;
	}

	DeviceInfo deviceInfo() {
		DeviceInfo info;
		bool present = false;
		@autoreleasepool {
			id<MTLDevice> device = MTLCreateSystemDefaultDevice();
			if (device != nil) {
				present = true;
				info.name = std::string([[device name] UTF8String]);
				info.recommendedMemoryBytes = static_cast<std::size_t>([device recommendedMaxWorkingSetSize]);
				info.unifiedMemory = [device hasUnifiedMemory] ? true : false;
			}
		}
		// Thrown outside the autorelease pool, so nothing is left to unwind through it.
		if (!present) {
			throw MetalError("metal::deviceInfo: No Metal device is available.");
		}
		return info;
	}

} // namespace metal
