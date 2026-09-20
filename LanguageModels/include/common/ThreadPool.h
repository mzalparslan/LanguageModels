#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/**
 * @brief Small fixed pool of worker threads with one operation, parallelFor(),
 * which splits a range of independent work into contiguous chunks.
 *
 * Built for the numeric loops of this library (zeroing, updating and
 * multiplying large tensors), where work per element is tiny: threads are
 * created once and reused, and a range too small to be worth splitting simply
 * runs on the calling thread.
 *
 * The calling thread takes the first chunk itself, so a pool of N workers
 * gives N + 1 threads of work. Chunks are contiguous and independent, which
 * lets callers keep the per-element order of operations unchanged and get
 * bit-identical results whatever the thread count.
 *
 * @remark parallelFor() may be called from several threads at once, but must
 * not be called from inside a parallelFor() body: every worker could then be
 * waiting on work that only a worker can run.
 */
class ThreadPool {
public:
	/**
	 * @param workerCount Number of background threads (0 makes a pool that
	 * runs everything on the caller).
	 */
	explicit ThreadPool(std::size_t workerCount) {
		for (std::size_t i = 0; i < workerCount; i++) {
			workers.emplace_back([this] { workerLoop(); });
		}
	}

	~ThreadPool() {
		{
			std::lock_guard<std::mutex> lock(queueMutex);
			stopping = true;
		}
		wakeWorkers.notify_all();
		for (std::thread& worker : workers) {
			worker.join();
		}
	}

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool& operator=(const ThreadPool&) = delete;

	/**
	 * @brief Most threads that can work on one parallelFor() call (the
	 * workers plus the caller).
	 */
	std::size_t maxThreads() const { return workers.size() + 1; }

	/**
	 * @brief Process-wide pool used by the library's *Parallel methods.
	 * Created on first use with at least 8 threads (or the hardware thread
	 * count, if larger); asking for more threads than the machine has is
	 * harmless, only slower.
	 */
	static ThreadPool& shared() {
		static ThreadPool pool(std::max<std::size_t>(std::thread::hardware_concurrency(), 8) - 1);
		return pool;
	}

	/**
	 * @brief Number of threads to use when the caller has no preference: the
	 * hardware thread count (at least 1).
	 */
	static std::size_t defaultThreadCount() {
		return std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
	}

	/**
	 * @brief Runs body(begin, end) over the range [0, count), split into
	 * contiguous chunks that run at the same time, and returns when every
	 * chunk is done.
	 *
	 * @param count Number of independent items.
	 * @param minPerChunk Fewest items worth giving to one thread; a range
	 * shorter than 2 * minPerChunk is not split.
	 * @param threads Most threads to use (1 runs everything on the caller).
	 * @param body Called once per chunk with its half-open item range.
	 * Chunks must not depend on each other.
	 * @throws Whatever a chunk threw (the first one, if several did), after
	 * all chunks have finished.
	 */
	void parallelFor(std::size_t count, std::size_t minPerChunk, std::size_t threads,
		const std::function<void(std::size_t, std::size_t)>& body) {
		if (count == 0) {
			return;
		}

		std::size_t chunks = std::min(threads, maxThreads());
		chunks = std::min(chunks, count / std::max<std::size_t>(minPerChunk, 1));
		if (chunks <= 1) {
			body(0, count);
			return;
		}

		// Shared with the queued tasks, which may still be finishing their
		// bookkeeping when the caller returns.
		auto state = std::make_shared<CallState>();
		state->remaining = chunks - 1;

		// Chunk c covers [count * c / chunks, count * (c + 1) / chunks).
		auto chunkBegin = [count, chunks](std::size_t c) { return count / chunks * c + std::min(c, count % chunks); };
		for (std::size_t c = 1; c < chunks; c++) {
			const std::size_t begin = chunkBegin(c);
			const std::size_t end = chunkBegin(c + 1);
			enqueue([state, &body, begin, end] {
				try {
					body(begin, end);
				}
				catch (...) {
					state->recordError(std::current_exception());
				}
				state->finishOne();
			});
		}

		try {
			body(0, chunkBegin(1));
		}
		catch (...) {
			state->recordError(std::current_exception());
		}

		std::unique_lock<std::mutex> lock(state->mutex);
		state->done.wait(lock, [&state] { return state->remaining == 0; });
		if (state->error) {
			std::rethrow_exception(state->error);
		}
	}

private:
	/**
	 * @brief Completion bookkeeping of one parallelFor() call.
	 */
	class CallState {
	public:
		std::mutex mutex;
		std::condition_variable done;
		std::size_t remaining = 0;
		std::exception_ptr error;

		void recordError(std::exception_ptr caught) {
			std::lock_guard<std::mutex> lock(mutex);
			if (!error) {
				error = caught;
			}
		}

		void finishOne() {
			std::lock_guard<std::mutex> lock(mutex);
			if (--remaining == 0) {
				done.notify_all();
			}
		}
	};

	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;
	std::mutex queueMutex;
	std::condition_variable wakeWorkers;
	bool stopping = false;

	void enqueue(std::function<void()> task) {
		{
			std::lock_guard<std::mutex> lock(queueMutex);
			tasks.push(std::move(task));
		}
		wakeWorkers.notify_one();
	}

	void workerLoop() {
		for (;;) {
			std::function<void()> task;
			{
				std::unique_lock<std::mutex> lock(queueMutex);
				wakeWorkers.wait(lock, [this] { return stopping || !tasks.empty(); });
				if (tasks.empty()) {
					return; // stopping, and nothing left to run
				}
				task = std::move(tasks.front());
				tasks.pop();
			}
			task();
		}
	}
};
