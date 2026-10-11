// Times one Mini Transformer training step the three ways the library can run it, and
// prints the result as a Markdown table:
//
//   Sequential  MiniTransformer::trainStep()                one thread
//   Parallel    MiniTransformer::trainStepMultipleThread()  every hardware thread
//   Device      MiniTransformer::trainStepDevice()          the GPU (CUDA or Metal), if any
//
// A step is one sentence pair (batch size 1) in float with Adam, at two vocabulary sizes:
// a small one, and the one of the full fra.txt translation data. The "Device" row says
// which GPU backend ran it, or why there was none.
//
//   make bench BUILD=release                   (Makefile, Linux / macOS)
//   LanguageModels.Benchmark.exe [--quick]     (--quick: fewer steps, for slow machines)

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "GpuVocabularyHead.h"
#include "MiniTransformer.h"

namespace {

	using Clock = std::chrono::steady_clock;
	using Model = MiniTransformer<float>;

	// Sentence pairs in the full fra.txt data: what an "epoch" below means.
	constexpr double fraPairs = 191351.0;

	const std::vector<std::size_t> source = { 3, 4, 5, 6, 7, 8 };
	const std::vector<std::size_t> decoderInput = { 1, 2, 3, 4, 5, 6, 7, 8 };
	const std::vector<std::size_t> labels = { 2, 3, 4, 5, 6, 7, 8, 0 };

	// Milliseconds per call, after some calls that are not counted (caches, GPU start-up).
	template <typename F>
	double msPerCall(int warmup, int runs, F&& step) {
		for (int i = 0; i < warmup; i++) {
			step(i + 1);
		}
		const Clock::time_point start = Clock::now();
		for (int i = 0; i < runs; i++) {
			step(warmup + i + 1);
		}
		return std::chrono::duration<double, std::milli>(Clock::now() - start).count() / runs;
	}

	std::string milliseconds(double ms) {
		std::ostringstream text;
		text << std::fixed << std::setprecision(ms < 10.0 ? 2 : 1) << ms << " ms";
		return text.str();
	}

	std::string ratio(double x) {
		std::ostringstream text;
		text << std::fixed << std::setprecision(x < 10.0 ? 1 : 0) << x << "x";
		return text.str();
	}

	// One epoch of the full data at this step time, as a rough human figure.
	std::string epoch(double ms) {
		const double minutes = ms * fraPairs / 1000.0 / 60.0;
		std::ostringstream text;
		text << "about " << std::fixed;
		if (minutes >= 90.0) {
			text << std::setprecision(1) << minutes / 60.0 << " hours";
		}
		else {
			text << std::setprecision(1) << minutes << " minutes";
		}
		return text.str();
	}

	void printTable(const std::string& title, bool withEpoch, std::size_t threads, double sequential,
		double parallel, bool deviceRan, double device, const std::string& deviceName) {
		std::cout << "\n### " << title << "\n\n";
		std::cout << "| Execution | per step | speedup |" << (withEpoch ? " one epoch (191,351 pairs) |" : "") << "\n";
		std::cout << "|---|---|---|" << (withEpoch ? "---|" : "") << "\n";

		auto row = [&](const std::string& name, double ms) {
			std::cout << "| " << name << " | " << milliseconds(ms) << " | " << ratio(sequential / ms) << " |"
				<< (withEpoch ? " " + epoch(ms) + " |" : "") << "\n";
		};
		row("`Sequential`: `trainStep`, one thread", sequential);
		row("`Parallel`: `trainStepMultipleThread`, " + std::to_string(threads) + " threads", parallel);
		if (deviceRan) {
			row("`Device`: `trainStepDevice`, " + deviceName, device);
		}
		else {
			std::cout << "| `Device`: `trainStepDevice` | not available | |" << (withEpoch ? " |" : "") << "\n";
		}
	}

	void benchmark(const std::string& title, std::size_t sourceVocab, std::size_t targetVocab, bool withEpoch, bool quick) {
		const int sequentialRuns = quick ? 5 : 15;
		const int parallelRuns = quick ? 10 : 30;
		const int deviceRuns = quick ? 60 : 200;
		const std::size_t threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
		std::cerr << "measuring " << title << " ..." << std::endl;

		double sequential = 0.0;
		{
			Model model(sourceVocab, targetVocab);
			sequential = msPerCall(2, sequentialRuns, [&](int step) {
				(void)model.trainStep(source, decoderInput, labels, 0.001f, UpdateRule::adam(step));
			});
		}

		double parallel = 0.0;
		{
			Model model(sourceVocab, targetVocab);
			parallel = msPerCall(3, parallelRuns, [&](int step) {
				(void)model.trainStepMultipleThread(source, decoderInput, labels, 0.001f, UpdateRule::adam(step), threads);
			});
		}

		double device = 0.0;
		const bool deviceRan = gpu::isAvailable<float>();
		std::string deviceName;
		if (deviceRan) {
			Model model(sourceVocab, targetVocab);
			gpu::VocabularyHead<float> head = model.createDeviceHead();
			deviceName = std::string(gpu::toString(head.backend())) + " GPU";
			device = msPerCall(10, deviceRuns, [&](int step) {
				(void)model.trainStepDevice(head, source, decoderInput, labels, 0.001f, UpdateRule::adam(step));
			});
		}

		printTable(title, withEpoch, threads, sequential, parallel, deviceRan, device, deviceName);
	}

} // namespace

int main(int argc, char** argv) {
	bool quick = false;
	for (int i = 1; i < argc; i++) {
		if (std::string(argv[i]) == "--quick") {
			quick = true;
		}
	}

	std::cout << "## Mini Transformer training step, float, Adam\n\n";
	std::cout << "- Hardware threads: " << std::thread::hardware_concurrency() << "\n";
	std::cout << "- GPU for `Device`: " << gpu::describeDevice<float>() << "\n";
	std::cout << "- One step is one sentence pair (batch size 1). Times are the mean over many steps.\n";

	benchmark("Vocabulary 3,000 source / 5,000 target", 3000, 5000, false, quick);
	benchmark("Vocabulary 16,097 source / 30,152 target (the full fra.txt)", 16097, 30152, true, quick);
	return 0;
}
