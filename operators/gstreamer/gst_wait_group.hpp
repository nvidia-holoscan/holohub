/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN__GSTREAMER__GST_WAIT_GROUP_HPP
#define HOLOSCAN__GSTREAMER__GST_WAIT_GROUP_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace holoscan {

/**
 * @brief GstWaitGroup for tracking active operations and waiting for completion.
 *
 * This class implements a synchronization primitive similar to Go's sync.WaitGroup.
 * It tracks the number of active operations using atomic operations and condition
 * variables. Operations call add() when starting and done() when completing.
 * The main thread can wait for all operations to complete via wait().
 *
 * Note: This is conceptually different from std::counting_semaphore (C++20),
 * which manages a pool of available resources (blocking when none are available)
 * rather than tracking active operations (waiting for count to reach zero).
 */
class GstWaitGroup {
 public:
  /**
   * @brief Construct a GstWaitGroup with initial count of 0.
   */
  GstWaitGroup() : count_(0) {}

  // Non-copyable and non-movable.
  GstWaitGroup(const GstWaitGroup&) = delete;
  GstWaitGroup& operator=(const GstWaitGroup&) = delete;
  GstWaitGroup(GstWaitGroup&&) = delete;
  GstWaitGroup& operator=(GstWaitGroup&&) = delete;

  /**
   * @brief Increment the counter.
   *
   * This should be called when an operation starts.
   * Similar to Go's sync.WaitGroup.Add(1).
   */
  void add() { count_.fetch_add(1, std::memory_order_relaxed); }

  /**
   * @brief Decrement the counter and notify waiters if it reaches zero.
   *
   * This should be called when an operation completes.
   * Similar to Go's sync.WaitGroup.Done().
   * 
   * @note Calling done() more times than add() results in undefined behavior.
   *       Use GstWaitGroupGuard to ensure proper pairing.
   */
  void done() {
    if (count_.fetch_sub(1, std::memory_order_release) == 1) {
      // We were the last active operation, notify all waiters.
      cv_.notify_all();
    }
  }

  /**
   * @brief Wait until the counter reaches zero.
   *
   * This blocks until all operations have completed (counter reaches 0).
   * Similar to Go's sync.WaitGroup.Wait().
   */
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return count_.load(std::memory_order_acquire) == 0; });
  }

  /**
   * @brief Get the current counter value.
   *
   * @return Current number of active operations.
   */
  size_t count() const { return count_.load(std::memory_order_acquire); }

 private:
  std::atomic<size_t> count_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

/**
 * @brief RAII guard for GstWaitGroup add/done.
 *
 * Automatically calls add() on construction and done() on destruction.
 * Use this to ensure done() is always called, even on exceptions.
 */
class GstWaitGroupGuard {
 public:
  explicit GstWaitGroupGuard(GstWaitGroup& wg) : wg_(wg) { wg_.add(); }

  ~GstWaitGroupGuard() { wg_.done(); }

  // Non-copyable and non-movable.
  GstWaitGroupGuard(const GstWaitGroupGuard&) = delete;
  GstWaitGroupGuard& operator=(const GstWaitGroupGuard&) = delete;
  GstWaitGroupGuard(GstWaitGroupGuard&&) = delete;
  GstWaitGroupGuard& operator=(GstWaitGroupGuard&&) = delete;

 private:
  GstWaitGroup& wg_;
};

}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST_WAIT_GROUP_HPP */
