#include "pch.h"
#include "ThreadPool.h"
#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

TEST(ThreadPoolTest, TheSharedPoolHasRoomForAtLeastEightThreads) {
    EXPECT_GE(ThreadPool::shared().maxThreads(), 8u);
    EXPECT_GE(ThreadPool::defaultThreadCount(), 1u);
}

TEST(ThreadPoolTest, EveryIndexIsVisitedExactlyOnce) {
    for (std::size_t count : { 1u, 2u, 7u, 100u, 1000u, 4099u }) {
        for (std::size_t threads : { 1u, 2u, 3u, 8u }) {
            std::vector<std::atomic<int>> visits(count);
            ThreadPool::shared().parallelFor(count, 1, threads, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; i++) {
                    visits[i]++;
                }
            });

            for (std::size_t i = 0; i < count; i++) {
                ASSERT_EQ(visits[i].load(), 1) << "count " << count << ", threads " << threads << ", index " << i;
            }
        }
    }
}

TEST(ThreadPoolTest, ChunksAreContiguousDisjointAndCoverTheRange) {
    std::mutex mutex;
    std::vector<std::pair<std::size_t, std::size_t>> ranges;

    ThreadPool::shared().parallelFor(1003, 1, 6, [&](std::size_t begin, std::size_t end) {
        std::lock_guard<std::mutex> lock(mutex);
        ranges.push_back({ begin, end });
    });

    std::sort(ranges.begin(), ranges.end());
    ASSERT_EQ(ranges.size(), 6u);
    EXPECT_EQ(ranges.front().first, 0u);
    EXPECT_EQ(ranges.back().second, 1003u);
    for (std::size_t i = 1; i < ranges.size(); i++) {
        EXPECT_EQ(ranges[i].first, ranges[i - 1].second);
        EXPECT_LT(ranges[i].first, ranges[i].second);
    }
}

TEST(ThreadPoolTest, AnEmptyRangeDoesNothing) {
    bool called = false;

    ThreadPool::shared().parallelFor(0, 1, 4, [&](std::size_t, std::size_t) { called = true; });

    EXPECT_FALSE(called);
}

TEST(ThreadPoolTest, OneThreadRunsEverythingOnTheCaller) {
    std::set<std::thread::id> ids;
    std::size_t chunks = 0;

    ThreadPool::shared().parallelFor(100000, 1, 1, [&](std::size_t begin, std::size_t end) {
        ids.insert(std::this_thread::get_id());
        chunks++;
        EXPECT_EQ(begin, 0u);
        EXPECT_EQ(end, 100000u);
    });

    EXPECT_EQ(chunks, 1u);
    EXPECT_EQ(ids, std::set<std::thread::id>{ std::this_thread::get_id() });
}

TEST(ThreadPoolTest, ARangeTooSmallToBeWorthSplittingRunsOnTheCaller) {
    std::size_t chunks = 0;

    // 100 items with at least 60 per chunk: only one chunk fits.
    ThreadPool::shared().parallelFor(100, 60, 8, [&](std::size_t, std::size_t) { chunks++; });

    EXPECT_EQ(chunks, 1u);
}

TEST(ThreadPoolTest, LargeRangesReallyUseMoreThanOneThread) {
    std::mutex mutex;
    std::set<std::thread::id> ids;

    ThreadPool::shared().parallelFor(1000, 1, 4, [&](std::size_t, std::size_t) {
        std::lock_guard<std::mutex> lock(mutex);
        ids.insert(std::this_thread::get_id());
    });

    // caller takes first chunk; others go to workers.
    EXPECT_GE(ids.size(), 2u);
    EXPECT_TRUE(ids.count(std::this_thread::get_id()));
}

TEST(ThreadPoolTest, ThreadsAskedForBeyondThePoolSizeAreCapped) {
    std::atomic<std::size_t> chunks{ 0 };

    ThreadPool::shared().parallelFor(100000, 1, 100000, [&](std::size_t, std::size_t) { chunks++; });

    EXPECT_EQ(chunks.load(), ThreadPool::shared().maxThreads());
}

TEST(ThreadPoolTest, AnExceptionInAChunkIsRethrownAfterEveryChunkFinished) {
    std::atomic<int> finished{ 0 };

    EXPECT_THROW(
        ThreadPool::shared().parallelFor(8, 1, 8, [&](std::size_t begin, std::size_t) {
            finished++;
            if (begin == 3) {
                throw std::runtime_error("chunk failed");
            }
        }),
        std::runtime_error);

    // Nothing is still running when parallelFor() gives up control.
    EXPECT_EQ(finished.load(), 8);
}

TEST(ThreadPoolTest, AnExceptionOnTheCallersOwnChunkIsRethrownToo) {
    EXPECT_THROW(
        ThreadPool::shared().parallelFor(8, 1, 8, [&](std::size_t begin, std::size_t) {
            if (begin == 0) {
                throw std::invalid_argument("first chunk failed");
            }
        }),
        std::invalid_argument);
}

TEST(ThreadPoolTest, ThePoolStaysUsableAfterAnException) {
    EXPECT_ANY_THROW(ThreadPool::shared().parallelFor(8, 1, 8, [](std::size_t, std::size_t) {
        throw std::runtime_error("boom");
    }));

    std::atomic<std::size_t> total{ 0 };
    ThreadPool::shared().parallelFor(1000, 1, 8, [&](std::size_t begin, std::size_t end) { total += end - begin; });
    EXPECT_EQ(total.load(), 1000u);
}

TEST(ThreadPoolTest, ManyCallsInARowGiveCorrectResults) {
    for (int round = 0; round < 500; round++) {
        std::vector<int> values(257, 1);
        ThreadPool::shared().parallelFor(values.size(), 1, 4, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; i++) {
                values[i] *= 2;
            }
        });
        ASSERT_EQ(std::accumulate(values.begin(), values.end(), 0), 2 * 257) << "round " << round;
    }
}

TEST(ThreadPoolTest, SeveralThreadsCanCallParallelForAtTheSameTime) {
    std::atomic<int> wrong{ 0 };
    std::vector<std::thread> callers;

    for (int caller = 0; caller < 4; caller++) {
        callers.emplace_back([&] {
            for (int round = 0; round < 100; round++) {
                std::vector<int> values(500, 0);
                ThreadPool::shared().parallelFor(values.size(), 1, 4, [&](std::size_t begin, std::size_t end) {
                    for (std::size_t i = begin; i < end; i++) {
                        values[i] = 1;
                    }
                });
                if (std::accumulate(values.begin(), values.end(), 0) != 500) {
                    wrong++;
                }
            }
        });
    }
    for (std::thread& caller : callers) {
        caller.join();
    }

    EXPECT_EQ(wrong.load(), 0);
}

TEST(ThreadPoolTest, APoolWithoutWorkersRunsEverythingOnTheCaller) {
    ThreadPool pool(0);
    std::size_t chunks = 0;

    pool.parallelFor(1000, 1, 8, [&](std::size_t begin, std::size_t end) {
        chunks++;
        EXPECT_EQ(begin, 0u);
        EXPECT_EQ(end, 1000u);
    });

    EXPECT_EQ(pool.maxThreads(), 1u);
    EXPECT_EQ(chunks, 1u);
}

TEST(ThreadPoolTest, ADestroyedPoolJoinsItsWorkers) {
    std::atomic<std::size_t> total{ 0 };
    {
        ThreadPool pool(3);
        pool.parallelFor(400, 1, 4, [&](std::size_t begin, std::size_t end) { total += end - begin; });
    }

    EXPECT_EQ(total.load(), 400u);
}
