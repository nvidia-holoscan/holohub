/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "api/rmax_apps_lib_api.h"

#include "rivermax_service/rmax_ipo_receiver_service.h"
#include "rivermax_mgr_impl/rivermax_chunk_consumer_ano.h"
#include <holoscan/logger/logger.hpp>

#define USE_BLOCKING_QUEUE 0
#define USE_BLOCKING_MEMPOOL 1

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace holoscan::advanced_network {

/**
 * @brief A non-blocking queue implementation.
 *
 * @tparam T Type of elements stored in the queue.
 */
template <typename T>
class NonBlockingQueue : public QueueInterface<T> {
  std::queue<T> queue_;
  mutable std::mutex mutex_;

 public:
  /**
   * @brief Enqueues an element into the queue.
   *
   * @param value The element to be enqueued.
   */
  void enqueue(const T& value) override {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(value);
  }

  /**
   * @brief Tries to dequeue an element from the queue.
   *
   * @param value Reference to store the dequeued element.
   * @return true if an element was dequeued, false otherwise.
   */
  bool try_dequeue(T& value) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) { return false; }
    value = queue_.front();
    queue_.pop();
    return true;
  }

  /**
   * @brief Tries to dequeue an element from the queue.
   *
   * @param value Reference to store the dequeued element.
   * @param timeout Timeout for the dequeue operation (ignored).
   * @return true if an element was dequeued, false otherwise.
   */
  bool try_dequeue(T& value, std::chrono::milliseconds timeout) override {
    return try_dequeue(value);
  }

  /**
   * @brief Gets the size of the queue.
   *
   * @return The number of elements in the queue.
   */
  size_t get_size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  /**
   * @brief Clears all elements from the queue.
   */
  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) { queue_.pop(); }
  }
};

/**
 * @brief A blocking queue implementation.
 *
 * @tparam T Type of elements stored in the queue.
 */
template <typename T>
class BlockingQueue : public QueueInterface<T> {
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;

 public:
  /**
   * @brief Enqueues an element into the queue.
   *
   * @param value The element to be enqueued.
   */
  void enqueue(const T& value) override {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(value);
    cond_.notify_one();
  }

  /**
   * @brief Tries to dequeue an element from the queue (blocks forever).
   *
   * @param value Reference to store the dequeued element.
   * @return true if an element was dequeued, false otherwise.
   */
  bool try_dequeue(T& value) override {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty(); });
    value = queue_.front();
    queue_.pop();
    return true;
  }

  /**
   * @brief Tries to dequeue an element from the queue.
   *
   * @param value Reference to store the dequeued element.
   * @param timeout Timeout for the dequeue operation.
   * @return true if an element was dequeued, false otherwise.
   */
  bool try_dequeue(T& value, std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) { return false; }
    value = queue_.front();
    queue_.pop();
    return true;
  }

  /**
   * @brief Gets the size of the queue.
   *
   * @return The number of elements in the queue.
   */
  size_t get_size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  /**
   * @brief Clears all elements from the queue.
   */
  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) { queue_.pop(); }
  }
};

/**
 * @brief Memory pool for managing bursts of packets.
 */
class AnoBurstsMemoryPool : public IAnoBurstsCollection {
 public:
  AnoBurstsMemoryPool() = delete;

  /**
   * @brief Constructor with initial queue size.
   *
   * @param size Initial size of the memory pool.
   * @param burst_handler Reference to the burst handler.
   * @param tag Tag for the burst.
   */
  AnoBurstsMemoryPool(size_t size, RivermaxBurst::BurstHandler& burst_handler, uint32_t tag);

  /**
   * @brief Destructor for the AnoBurstsMemoryPool class.
   *
   * Frees all bursts in the queue and clears the burst map.
   */
  ~AnoBurstsMemoryPool();

  bool enqueue_burst(std::shared_ptr<RivermaxBurst> burst) override;
  bool enqueue_burst(RivermaxBurst* burst);
  std::shared_ptr<RivermaxBurst> dequeue_burst() override;
  size_t available_bursts() override { return m_queue->get_size(); };
  bool empty() override { return m_queue->get_size() == 0; };

 private:
  std::unique_ptr<QueueInterface<std::shared_ptr<RivermaxBurst>>> m_queue;
  std::map<uint16_t, std::shared_ptr<RivermaxBurst>> m_burst_map;
  size_t m_initial_size;
  mutable std::mutex m_burst_map_mutex;
  mutable std::mutex m_queue_mutex;
  uint32_t m_bursts_tag = 0;
  RivermaxBurst::BurstHandler& m_burst_handler;
};

AnoBurstsMemoryPool::AnoBurstsMemoryPool(size_t size, RivermaxBurst::BurstHandler& burst_handler,
                                         uint32_t tag)
    : m_initial_size(size), m_bursts_tag(tag), m_burst_handler(burst_handler) {
#if USE_BLOCKING_MEMPOOL
  m_queue = std::make_unique<BlockingQueue<std::shared_ptr<RivermaxBurst>>>();
#else
  m_queue = std::make_unique<NonBlockingQueue<std::shared_ptr<RivermaxBurst>>>();
#endif

  for (uint16_t i = 0; i < size; i++) {
    auto burst = m_burst_handler.create_burst(i);
    m_queue->enqueue(burst);
    m_burst_map[i] = burst;
  }
}

bool AnoBurstsMemoryPool::enqueue_burst(RivermaxBurst* burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return false;
  }

  uint16_t burst_id = burst->get_burst_id();

  std::lock_guard<std::mutex> lock(m_burst_map_mutex);
  auto it = m_burst_map.find(burst_id);
  if (it != m_burst_map.end()) {
    std::shared_ptr<RivermaxBurst> cur_burst = it->second;
    return enqueue_burst(cur_burst);
  } else {
    HOLOSCAN_LOG_ERROR("Invalid burst ID: {}", burst_id);
    return false;
  }
}

bool AnoBurstsMemoryPool::enqueue_burst(std::shared_ptr<RivermaxBurst> burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return false;
  }

  std::lock_guard<std::mutex> lock(m_queue_mutex);

  if (m_queue->get_size() < m_initial_size) {
    auto burst_tag = burst->get_burst_tag();
    if (m_bursts_tag != burst_tag) {
      HOLOSCAN_LOG_ERROR("Invalid burst tag");
      return false;
    }
    m_queue->enqueue(burst);
    return true;
  } else {
    HOLOSCAN_LOG_ERROR("Burst pool is full burst_pool_tag {}", m_bursts_tag);
  }
  return false;
}

std::shared_ptr<RivermaxBurst> AnoBurstsMemoryPool::dequeue_burst() {
  std::shared_ptr<RivermaxBurst> burst;

  if (m_queue->try_dequeue(burst,
                           std::chrono::milliseconds(RxBurstsManager::GET_BURST_TIMEOUT_MS))) {
    return burst;
  }
  return nullptr;
}

AnoBurstsMemoryPool::~AnoBurstsMemoryPool() {
  std::shared_ptr<RivermaxBurst> burst;

  while (m_queue->get_size() > 0 && m_queue->try_dequeue(burst)) {
    m_burst_handler.delete_burst(burst);
  }
  std::lock_guard<std::mutex> lock(m_burst_map_mutex);
  m_burst_map.clear();
}

AnoBurstsQueue::AnoBurstsQueue() {
#if USE_BLOCKING_QUEUE
  m_queue = std::make_unique<BlockingQueue<std::shared_ptr<RivermaxBurst>>>();
#else
  m_queue = std::make_unique<NonBlockingQueue<std::shared_ptr<RivermaxBurst>>>();
#endif
}

bool AnoBurstsQueue::enqueue_burst(std::shared_ptr<RivermaxBurst> burst) {
  m_queue->enqueue(burst);
  return true;
}

void AnoBurstsQueue::clear() {
  m_queue->clear();
}

std::shared_ptr<RivermaxBurst> AnoBurstsQueue::dequeue_burst() {
  std::shared_ptr<RivermaxBurst> burst;

  if (m_queue->try_dequeue(burst,
                           std::chrono::milliseconds(RxBurstsManager::GET_BURST_TIMEOUT_MS))) {
    return burst;
  }
  return nullptr;
}

RivermaxBurst::BurstHandler::BurstHandler(bool send_packet_ext_info, int port_id, int queue_id,
                                      bool gpu_direct)
    : m_send_packet_ext_info(send_packet_ext_info),
      m_port_id(port_id),
      m_queue_id(queue_id),
      m_gpu_direct(gpu_direct) {
  const uint32_t burst_tag = burst_tag_from_port_and_queue_id(port_id, queue_id);

  m_burst_info.tag = burst_tag;
  m_burst_info.burst_flags =
      (m_send_packet_ext_info ? BurstFlags::INFO_PER_PACKET : BurstFlags::FLAGS_NONE);
  m_burst_info.burst_id = 0;
  m_burst_info.hds_on = false;
  m_burst_info.header_on_cpu = false;
  m_burst_info.payload_on_cpu = false;
  m_burst_info.header_stride_size = 0;
  m_burst_info.payload_stride_size = 0;
  m_burst_info.header_seg_idx = 0;
  m_burst_info.payload_seg_idx = 0;
}

std::shared_ptr<RivermaxBurst> RivermaxBurst::BurstHandler::create_burst(uint16_t burst_id) {
  std::shared_ptr<RivermaxBurst> burst(new RivermaxBurst(m_port_id, m_queue_id, MAX_PKT_IN_BURST));

  if (m_send_packet_ext_info) {
    burst->pkt_extra_info = reinterpret_cast<void**>(new RivermaxPacketExtendedInfo*[MAX_PKT_IN_BURST]);
    for (int j = 0; j < MAX_PKT_IN_BURST; j++) {
      burst->pkt_extra_info[j] = reinterpret_cast<void*>(new RivermaxPacketExtendedInfo());
    }
  }

  burst->pkts[0] = new void*[MAX_PKT_IN_BURST];
  burst->pkts[1] = new void*[MAX_PKT_IN_BURST];
  burst->pkt_lens[0] = new uint32_t[MAX_PKT_IN_BURST];
  burst->pkt_lens[1] = new uint32_t[MAX_PKT_IN_BURST];
  std::memset(burst->pkt_lens[0], 0, MAX_PKT_IN_BURST * sizeof(uint32_t));
  std::memset(burst->pkt_lens[1], 0, MAX_PKT_IN_BURST * sizeof(uint32_t));

  m_burst_info.burst_id = burst_id;
  std::memcpy(burst->get_burst_info(), &m_burst_info, sizeof(m_burst_info));
  return burst;
}

void RivermaxBurst::BurstHandler::delete_burst(std::shared_ptr<RivermaxBurst> burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return;
  }

  if (m_send_packet_ext_info && burst->pkt_extra_info != nullptr) {
    for (int i = 0; i < MAX_PKT_IN_BURST; i++) {
      if (burst->pkt_extra_info[i] != nullptr) {
        delete reinterpret_cast<RivermaxPacketExtendedInfo*>(burst->pkt_extra_info[i]);
        burst->pkt_extra_info[i] = nullptr;
      }
    }
    delete[] burst->pkt_extra_info;
    burst->pkt_extra_info = nullptr;
  }

  delete[] burst->pkts[0];
  burst->pkts[0] = nullptr;
  delete[] burst->pkts[1];
  burst->pkts[1] = nullptr;
  delete[] burst->pkt_lens[0];
  burst->pkt_lens[0] = nullptr;
  delete[] burst->pkt_lens[1];
  burst->pkt_lens[1] = nullptr;
}

void RxBurstsManager::rx_burst_done(RivermaxBurst* burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return;
  }

  IAnoBurstsCollection* basePtr = m_rx_bursts_mempool.get();

  AnoBurstsMemoryPool* derivedPtr = dynamic_cast<AnoBurstsMemoryPool*>(basePtr);

  if (derivedPtr) {
    bool rc = derivedPtr->enqueue_burst(burst);
    if (!rc) {
      HOLOSCAN_LOG_ERROR("Failed to push burst back to the pool. Port_id {}:{}, queue_id {}:{}",
                         burst->get_port_id(),
                         m_port_id,
                         burst->get_queue_id(),
                         m_queue_id);
    }
  } else {
    HOLOSCAN_LOG_ERROR("Failed to push burst back to the pool, cast failed");
  }
}

RxBurstsManager::RxBurstsManager(bool send_packet_ext_info, int port_id, int queue_id,
                                 uint16_t burst_out_size, int gpu_id,
                                 std::shared_ptr<IAnoBurstsCollection> rx_bursts_out_queue)
    : m_send_packet_ext_info(send_packet_ext_info),
      m_port_id(port_id),
      m_queue_id(queue_id),
      m_burst_out_size(burst_out_size),
      m_gpu_id(gpu_id),
      m_rx_bursts_out_queue(rx_bursts_out_queue),
      m_burst_handler(std::make_unique<RivermaxBurst::BurstHandler>(
          send_packet_ext_info, port_id, queue_id, gpu_id != INVALID_GPU_ID)) {
  const uint32_t burst_tag = RivermaxBurst::burst_tag_from_port_and_queue_id(port_id, queue_id);
  m_gpu_direct = (m_gpu_id != INVALID_GPU_ID);

  m_rx_bursts_mempool =
      std::make_unique<AnoBurstsMemoryPool>(DEFAULT_NUM_RX_BURSTS, *m_burst_handler, burst_tag);

  if (!m_rx_bursts_out_queue) {
    m_rx_bursts_out_queue = std::make_shared<AnoBurstsQueue>();
    m_using_shared_out_queue = false;
  }

  if (m_burst_out_size > RivermaxBurst::MAX_PKT_IN_BURST || m_burst_out_size == 0)
    m_burst_out_size = RivermaxBurst::MAX_PKT_IN_BURST;
}

RxBurstsManager::~RxBurstsManager() {
  if (m_using_shared_out_queue) { return; }

  std::shared_ptr<RivermaxBurst> burst;
  // Get all bursts from the queue and return them to the memory pool
  while (m_rx_bursts_out_queue->available_bursts() > 0) {
    burst = m_rx_bursts_out_queue->dequeue_burst();
    if (burst == nullptr) break;
    m_rx_bursts_mempool->enqueue_burst(burst);
  }
}

};  // namespace holoscan::advanced_network
