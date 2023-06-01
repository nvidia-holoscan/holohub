/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>

namespace yuan {
namespace holoscan {

template <typename T>
class threadsafe_queue_t {
  std::deque<T> m_Queue;
  mutable std::mutex m_Mutex;
  std::condition_variable m_CV;
  bool m_bIsQuit;
  bool m_bHasSignal;

  // Moved out of public interface to prevent races between this
  // and pop().
  bool empty() const { return m_Queue.empty(); }

 public:
  threadsafe_queue_t() : m_bIsQuit(false), m_bHasSignal(true){};
  threadsafe_queue_t(const threadsafe_queue_t<T>&) = delete;
  threadsafe_queue_t& operator=(const threadsafe_queue_t<T>&) = delete;

  threadsafe_queue_t(threadsafe_queue_t<T>&& other) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_Queue = std::move(other.m_Queue);
  }

  virtual ~threadsafe_queue_t() {}

  unsigned long size() const {
    std::unique_lock<std::mutex> lock(m_Mutex);
    return m_Queue.size();
  }

  bool pop(T& item) {
    std::unique_lock<std::mutex> lock(m_Mutex);
    if (m_Queue.empty()) { return false; }
    item = m_Queue.front();
    m_Queue.pop_front();
    return true;
  }

  bool pop_block(T& item) {
    std::unique_lock<std::mutex> lock(m_Mutex);
    m_CV.wait(lock, [&] { return (!m_Queue.empty() || m_bIsQuit || !m_bHasSignal); });
    if (m_Queue.empty() || m_bIsQuit || !m_bHasSignal) { return false; }
    item = m_Queue.front();
    m_Queue.pop_front();
    return true;
  }

  void push(const T& item) {
    std::unique_lock<std::mutex> lock(m_Mutex);
    m_Queue.push_back(item);
    m_CV.notify_all();
  }

  void push_and_drop(const T& item, std::function<void(T)> drop_function) {
    std::unique_lock<std::mutex> lock(m_Mutex);
    while (m_Queue.size() > 0) {
      T drop = m_Queue.front();
      drop_function(drop);
      m_Queue.pop_front();
    }
    m_Queue.push_back(item);
    m_CV.notify_all();
  }

  void signal(bool hasSignal) {
    std::unique_lock<std::mutex> lock(m_Mutex);
    m_bHasSignal = hasSignal;
    m_CV.notify_all();
  }

  void quit() {
    std::unique_lock<std::mutex> lock(m_Mutex);
    m_bIsQuit = true;
    m_CV.notify_all();
  }

  void reset() {
    std::unique_lock<std::mutex> lock(m_Mutex);
    m_bIsQuit = false;
    m_Queue.clear();
  }
};

}  // namespace holoscan
}  // namespace yuan
