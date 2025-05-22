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
#include <atomic>
#include <condition_variable>
#include <chrono>

#include <holoscan/logger/logger.hpp>

#include "rivermax_mgr_impl/rivermax_chunk_consumer_ano.h"

#define USE_BLOCKING_QUEUE 0
#define USE_BLOCKING_MEMPOOL 1

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
  std::atomic<bool> stop_{false};

 public:
  void enqueue(const T& value) override {
    if (stop_) { return; }
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(value);
  }

  bool try_dequeue(T& value) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty() || stop_) { return false; }
    value = queue_.front();
    queue_.pop();
    return true;
  }

  bool try_dequeue(T& value, std::chrono::milliseconds timeout) override {
    return try_dequeue(value);
  }

  size_t get_size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) { queue_.pop(); }
  }

  void stop() override {
    stop_ = true;
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
  std::atomic<bool> stop_{false};

 public:
  void enqueue(const T& value) override {
    if (stop_) { return; }
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(value);
    cond_.notify_one();
  }

  bool try_dequeue(T& value) override {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty() || stop_; });
    if (stop_) { return false; }
    value = queue_.front();
    queue_.pop();
    return true;
  }

  bool try_dequeue(T& value, std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty() || stop_; })) {
       return false;
    }
    if (stop_) { return false; }
    value = queue_.front();
    queue_.pop();
    return true;
  }

  size_t get_size() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) { queue_.pop(); }
  }

  void stop() override {
      stop_ = true;
      cond_.notify_all();
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
  size_t available_bursts() override { return queue_->get_size(); };
  bool empty() override { return queue_->get_size() == 0; };
  void stop() override { queue_->stop(); }

 private:
  std::unique_ptr<QueueInterface<std::shared_ptr<RivermaxBurst>>> queue_;
  std::map<uint16_t, std::shared_ptr<RivermaxBurst>> burst_map_;
  size_t initial_size_;
  mutable std::mutex burst_map_mutex_;
  mutable std::mutex queue_mutex_;
  uint32_t bursts_tag_ = 0;
  RivermaxBurst::BurstHandler& burst_handler_;
};

AnoBurstsMemoryPool::AnoBurstsMemoryPool(size_t size, RivermaxBurst::BurstHandler& burst_handler,
                                         uint32_t tag)
    : initial_size_(size), bursts_tag_(tag), burst_handler_(burst_handler) {
#if USE_BLOCKING_MEMPOOL
  queue_ = std::make_unique<BlockingQueue<std::shared_ptr<RivermaxBurst>>>();
#else
  queue_ = std::make_unique<NonBlockingQueue<std::shared_ptr<RivermaxBurst>>>();
#endif

  for (uint16_t i = 0; i < size; i++) {
    auto burst = burst_handler_.create_burst(i);
    queue_->enqueue(burst);
    burst_map_[i] = burst;
  }
}

bool AnoBurstsMemoryPool::enqueue_burst(RivermaxBurst* burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return false;
  }

  uint16_t burst_id = burst->get_burst_id();

  std::lock_guard<std::mutex> lock(burst_map_mutex_);
  auto it = burst_map_.find(burst_id);
  if (it != burst_map_.end()) {
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

  std::lock_guard<std::mutex> lock(queue_mutex_);

  if (queue_->get_size() < initial_size_) {
    auto burst_tag = burst->get_burst_tag();
    if (bursts_tag_ != burst_tag) {
      HOLOSCAN_LOG_ERROR("Invalid burst tag");
      return false;
    }
    queue_->enqueue(burst);
    return true;
  } else {
    HOLOSCAN_LOG_ERROR("Burst pool is full burst_pool_tag {}", bursts_tag_);
  }
  return false;
}

std::shared_ptr<RivermaxBurst> AnoBurstsMemoryPool::dequeue_burst() {
  std::shared_ptr<RivermaxBurst> burst;

  if (queue_->try_dequeue(burst,
                           std::chrono::milliseconds(RxBurstsManager::GET_BURST_TIMEOUT_MS))) {
    return burst;
  }
  return nullptr;
}

AnoBurstsMemoryPool::~AnoBurstsMemoryPool() {
  std::shared_ptr<RivermaxBurst> burst;

  while (queue_->get_size() > 0 && queue_->try_dequeue(burst)) {
    burst_handler_.delete_burst(burst);
  }
  std::lock_guard<std::mutex> lock(burst_map_mutex_);
  burst_map_.clear();
}

AnoBurstsQueue::AnoBurstsQueue() {
#if USE_BLOCKING_QUEUE
  queue_ = std::make_unique<BlockingQueue<std::shared_ptr<RivermaxBurst>>>();
#else
  queue_ = std::make_unique<NonBlockingQueue<std::shared_ptr<RivermaxBurst>>>();
#endif
}

bool AnoBurstsQueue::enqueue_burst(std::shared_ptr<RivermaxBurst> burst) {
  queue_->enqueue(burst);
  return true;
}

void AnoBurstsQueue::clear() {
  queue_->clear();
}

void AnoBurstsQueue::stop() {
  queue_->stop();
}

std::shared_ptr<RivermaxBurst> AnoBurstsQueue::dequeue_burst() {
  std::shared_ptr<RivermaxBurst> burst;

  if (queue_->try_dequeue(burst,
                           std::chrono::milliseconds(RxBurstsManager::GET_BURST_TIMEOUT_MS))) {
    return burst;
  }
  return nullptr;
}

RivermaxBurst::BurstHandler::BurstHandler(bool send_packet_ext_info, int port_id, int queue_id,
                                          bool gpu_direct)
    : send_packet_ext_info_(send_packet_ext_info),
      port_id_(port_id),
      queue_id_(queue_id),
      gpu_direct_(gpu_direct) {
  const uint32_t burst_tag = burst_tag_from_port_and_queue_id(port_id, queue_id);

  burst_info_.tag = burst_tag;
  burst_info_.burst_id = 0;
  burst_info_.hds_on = false;
  burst_info_.header_on_cpu = false;
  burst_info_.payload_on_cpu = false;
  burst_info_.header_stride_size = 0;
  burst_info_.payload_stride_size = 0;
  burst_info_.header_seg_idx = 0;
  burst_info_.payload_seg_idx = 0;
}

std::shared_ptr<RivermaxBurst> RivermaxBurst::BurstHandler::create_burst(uint16_t burst_id) {
  std::shared_ptr<RivermaxBurst> burst(new RivermaxBurst(port_id_, queue_id_, MAX_PKT_IN_BURST));

  if (send_packet_ext_info_) {
    burst->hdr.hdr.burst_flags = BurstFlags::INFO_PER_PACKET;
    burst->pkt_extra_info =
        reinterpret_cast<void**>(new RivermaxPacketExtendedInfo*[MAX_PKT_IN_BURST]);
    for (int j = 0; j < MAX_PKT_IN_BURST; j++) {
      burst->pkt_extra_info[j] = reinterpret_cast<void*>(new RivermaxPacketExtendedInfo());
    }
  } else {
    burst->hdr.hdr.burst_flags = BurstFlags::FLAGS_NONE;
    burst->pkt_extra_info = nullptr;
  }

  burst->pkts[0] = new void*[MAX_PKT_IN_BURST];
  burst->pkts[1] = new void*[MAX_PKT_IN_BURST];
  burst->pkt_lens[0] = new uint32_t[MAX_PKT_IN_BURST];
  burst->pkt_lens[1] = new uint32_t[MAX_PKT_IN_BURST];
  std::memset(burst->pkt_lens[0], 0, MAX_PKT_IN_BURST * sizeof(uint32_t));
  std::memset(burst->pkt_lens[1], 0, MAX_PKT_IN_BURST * sizeof(uint32_t));

  burst_info_.burst_id = burst_id;
  std::memcpy(burst->get_burst_info(), &burst_info_, sizeof(burst_info_));
  return burst;
}

void RivermaxBurst::BurstHandler::delete_burst(std::shared_ptr<RivermaxBurst> burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return;
  }

  if (send_packet_ext_info_ && burst->pkt_extra_info != nullptr) {
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

  IAnoBurstsCollection* basePtr = rx_bursts_mempool_.get();

  AnoBurstsMemoryPool* derivedPtr = dynamic_cast<AnoBurstsMemoryPool*>(basePtr);

  if (derivedPtr) {
    bool rc = derivedPtr->enqueue_burst(burst);
    if (!rc) {
      HOLOSCAN_LOG_ERROR("Failed to push burst back to the pool. Port_id {}:{}, queue_id {}:{}",
                         burst->get_port_id(),
                         port_id_,
                         burst->get_queue_id(),
                         queue_id_);
    }
  } else {
    HOLOSCAN_LOG_ERROR("Failed to push burst back to the pool, cast failed");
  }
}

RxBurstsManager::RxBurstsManager(bool send_packet_ext_info, int port_id, int queue_id,
                                 uint16_t burst_out_size, int gpu_id,
                                 std::shared_ptr<IAnoBurstsCollection> rx_bursts_out_queue)
    : send_packet_ext_info_(send_packet_ext_info),
      port_id_(port_id),
      queue_id_(queue_id),
      burst_out_size_(burst_out_size),
      gpu_id_(gpu_id),
      rx_bursts_out_queue_(rx_bursts_out_queue),
      burst_handler_(std::make_unique<RivermaxBurst::BurstHandler>(
          send_packet_ext_info, port_id, queue_id, gpu_id != INVALID_GPU_ID)) {
  const uint32_t burst_tag = RivermaxBurst::burst_tag_from_port_and_queue_id(port_id, queue_id);
  gpu_direct_ = (gpu_id_ != INVALID_GPU_ID);

  rx_bursts_mempool_ =
      std::make_unique<AnoBurstsMemoryPool>(DEFAULT_NUM_RX_BURSTS, *burst_handler_, burst_tag);

  if (!rx_bursts_out_queue_) {
    rx_bursts_out_queue_ = std::make_shared<AnoBurstsQueue>();
    using_shared_out_queue_ = false;
  }

  if (burst_out_size_ > RivermaxBurst::MAX_PKT_IN_BURST || burst_out_size_ == 0)
  burst_out_size_ = RivermaxBurst::MAX_PKT_IN_BURST;
}

RxBurstsManager::~RxBurstsManager() {
  if (using_shared_out_queue_) { return; }

  std::shared_ptr<RivermaxBurst> burst;
  // Get all bursts from the queue and return them to the memory pool
  while (rx_bursts_out_queue_->available_bursts() > 0) {
    burst = rx_bursts_out_queue_->dequeue_burst();
    if (burst == nullptr) break;
    rx_bursts_mempool_->enqueue_burst(burst);
  }
}

};  // namespace holoscan::advanced_network
