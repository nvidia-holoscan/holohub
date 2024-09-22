/*
 * Copyright © 2017-2024 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of Nvidia Corporation and its affiliates
 * (the "Company") and all right, title, and interest in and to the software
 * product, including all associated intellectual property rights, are and
 * shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 */

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <rivermax_api.h>
#include "api/rmax_apps_lib_api.h"
#include "rmax_service/rmax_ipo_receiver_service.h"
#include "rmax_mgr_impl/rmax_chunk_consumer_ano.h"
#include <holoscan/logger/logger.hpp>

#define USE_BLOCKING_QUEUE 0
#define USE_BLOCKING_MEMPOOL 1

using namespace ral::lib::core;
using namespace ral::lib::services;

namespace holoscan::ops {

/**
 * @brief A non-blocking queue implementation.
 *
 * @tparam T Type of elements stored in the queue.
 */
template <typename T>
class NonBlockingQueue : public QueueInterface<T> {
  std::queue<T> queue_;       ///< Internal queue to store elements.
  mutable std::mutex mutex_;  ///< Mutex to protect access to the queue.

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
  std::queue<T> queue_;           ///< Internal queue to store elements.
  mutable std::mutex mutex_;      ///< Mutex to protect access to the queue.
  std::condition_variable cond_;  ///< Condition variable for blocking operations.

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
   * @param rx_burst_manager Reference to the RX burst manager.
   * @param tag Tag for the burst.
   */
  AnoBurstsMemoryPool(size_t size, RxBurstsManager& rx_burst_manager, uint32_t tag);

  /**
   * @brief Destructor to clean up resources.
   */
  ~AnoBurstsMemoryPool();

  bool enqueue_burst(std::shared_ptr<AdvNetBurstParams> burst) override;
  bool enqueue_burst(AdvNetBurstParams* burst);
  std::shared_ptr<AdvNetBurstParams> dequeue_burst() override;
  size_t available_bursts() override { return m_queue->get_size(); };
  bool empty() override { return m_queue->get_size() == 0; };

 private:
  std::unique_ptr<QueueInterface<std::shared_ptr<AdvNetBurstParams>>>
      m_queue;  ///< Queue to manage bursts.
  std::map<uint16_t, std::shared_ptr<AdvNetBurstParams>>
      m_burst_map;        ///< Map to store bursts by ID.
  size_t m_initial_size;  ///< Initial size of the memory pool.
  mutable std::mutex m_burst_map_mutex;  ///< Mutex to protect access to the burst map.
  mutable std::mutex m_queue_mutex;      ///< Mutex to protect access to the queue.
  uint32_t m_bursts_tag = 0;  ///< Tag for the bursts.
  RxBurstsManager& m_rx_bursts_manager;  ///< Reference to the RX bursts manager.
};

/**
 * @brief Constructor for AnoBurstsMemoryPool.
 *
 * @param size Initial size of the memory pool.
 * @param rx_burst_manager Reference to the RX burst manager.
 * @param tag Tag for the burst.
 */
AnoBurstsMemoryPool::AnoBurstsMemoryPool(size_t size, RxBurstsManager& rx_burst_manager,
                                         uint32_t tag)
    : m_initial_size(size), m_bursts_tag(tag), m_rx_bursts_manager(rx_burst_manager) {
#if USE_BLOCKING_MEMPOOL
  m_queue = std::make_unique<BlockingQueue<std::shared_ptr<AdvNetBurstParams>>>();
#else
  m_queue = std::make_unique<NonBlockingQueue<std::shared_ptr<AdvNetBurstParams>>>();
#endif

  // Initialize bursts and add them to the queue
  for (uint16_t i = 0; i < size; i++) {
    auto burst = m_rx_bursts_manager.create_burst(i);
    m_queue->enqueue(burst);
    m_burst_map[i] = burst;
  }
}

/**
 * @brief Puts a burst back into the memory pool.
 *
 * @param burst Pointer to the burst to be put back.
 * @return true if the burst was successfully put back, false otherwise.
 */
bool AnoBurstsMemoryPool::enqueue_burst(AdvNetBurstParams* burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return false;
  }

  // Read burst info
  uint16_t burst_id = m_rx_bursts_manager.get_burst_id(burst);

  std::lock_guard<std::mutex> lock(m_burst_map_mutex);
  auto it = m_burst_map.find(burst_id);
  if (it != m_burst_map.end()) {
    std::shared_ptr<AdvNetBurstParams> cur_burst = it->second;
    return enqueue_burst(cur_burst);
  } else {
    HOLOSCAN_LOG_ERROR("Invalid burst ID: {}", burst_id);
    return false;
  }
}

/**
 * @brief Puts a burst back into the memory pool.
 *
 * @param burst Shared pointer to the burst to be put back.
 * @return true if the burst was successfully put back, false otherwise.
 */
bool AnoBurstsMemoryPool::enqueue_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  // Lock the mutex to ensure thread safety
  std::lock_guard<std::mutex> lock(m_queue_mutex);

  if (m_queue->get_size() < m_initial_size) {
    auto burst_tag = m_rx_bursts_manager.get_burst_tag(burst.get());
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

/**
 * @brief Retrieves a burst from the memory pool.
 *
 * @return A shared pointer to the retrieved burst, or nullptr if no burst is available.
 */
std::shared_ptr<AdvNetBurstParams> AnoBurstsMemoryPool::dequeue_burst() {
  std::shared_ptr<AdvNetBurstParams> burst;
  // Try to dequeue a burst from the queue with a timeout
  if (m_queue->try_dequeue(burst,
                           std::chrono::milliseconds(RxBurstsManager::GET_BURST_TIMEOUT_MS))) {
    return burst;
  }
  return nullptr;
}

/**
 * @brief Destructor for the AnoBurstsMemoryPool class.
 *
 * Frees all bursts in the queue and clears the burst map.
 */
AnoBurstsMemoryPool::~AnoBurstsMemoryPool() {
  std::shared_ptr<AdvNetBurstParams> burst;
  // Dequeue and delete all bursts in the queue
  while (m_queue->get_size() > 0 && m_queue->try_dequeue(burst)) {
    m_rx_bursts_manager.delete_burst(burst);
  }
  std::lock_guard<std::mutex> lock(m_burst_map_mutex);
  m_burst_map.clear();
}

/**
 * @brief Constructor for the AnoBurstsQueue class.
 *
 * Initializes the queue based on the USE_BLOCKING_QUEUE macro.
 */
AnoBurstsQueue::AnoBurstsQueue() {
#if USE_BLOCKING_QUEUE
  m_queue = std::make_unique<BlockingQueue<std::shared_ptr<AdvNetBurstParams>>>();
#else
  m_queue = std::make_unique<NonBlockingQueue<std::shared_ptr<AdvNetBurstParams>>>();
#endif
}

/**
 * @brief Enqueues a burst into the queue.
 *
 * @param burst A shared pointer to the burst to be enqueued.
 * @return True if the burst was successfully enqueued.
 */
bool AnoBurstsQueue::enqueue_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  m_queue->enqueue(burst);
  return true;
}

/**
 * @brief Clears all bursts from the queue.
 */
void AnoBurstsQueue::clear() {
  m_queue->clear();
}

/**
 * @brief Retrieves a burst from the queue.
 *
 * @return A shared pointer to the retrieved burst, or nullptr if no burst is available.
 */
std::shared_ptr<AdvNetBurstParams> AnoBurstsQueue::dequeue_burst() {
  std::shared_ptr<AdvNetBurstParams> burst;
  // Try to dequeue a burst from the queue with a timeout
  if (m_queue->try_dequeue(burst,
                           std::chrono::milliseconds(RxBurstsManager::GET_BURST_TIMEOUT_MS))) {
    return burst;
  }
  return nullptr;
}

/**
 * @brief Marks a burst as done and returns it to the memory pool.
 *
 * @param burst Pointer to the burst that is done.
 */
void RxBurstsManager::rx_burst_done(AdvNetBurstParams* burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return;
  }

  // Get the base pointer to the burst pool
  IAnoBurstsCollection* basePtr = m_rx_bursts_mempool.get();

  // Use dynamic_cast to safely cast to the derived class
  AnoBurstsMemoryPool* derivedPtr = dynamic_cast<AnoBurstsMemoryPool*>(basePtr);

  if (derivedPtr) {
    // Return the burst to the memory pool
    bool rc = derivedPtr->enqueue_burst(burst);
    if (!rc) {
      HOLOSCAN_LOG_ERROR("Failed to push burst back to the pool. Port_id {}:{}, queue_id {}:{}",
                         burst->hdr.hdr.port_id,
                         m_port_id,
                         burst->hdr.hdr.q_id,
                         m_queue_id);
    }
  } else {
    HOLOSCAN_LOG_ERROR("Failed to push burst back to the pool, cast failed");
  }
}

/**
 * @brief Initializes a burst with the given burst ID.
 *
 * @param burst_id ID of the burst.
 * @return Shared pointer to the initialized burst.
 */
std::shared_ptr<AdvNetBurstParams> RxBurstsManager::create_burst(uint16_t burst_id) {
  auto burst = std::make_shared<AdvNetBurstParams>();

  if (m_send_packet_info) {
    // Allocate memory for the packets
    burst->pkt_extra_info = reinterpret_cast<void**>(new RmaxPacketExtendedInfo*[MAX_PKT_BURST]);
    for (int j = 0; j < MAX_PKT_BURST; j++) {
      burst->pkt_extra_info[j] = reinterpret_cast<void*>(new RmaxPacketExtendedInfo());
    }
  }

  burst->pkts[0] = new void*[MAX_PKT_BURST];
  burst->pkts[1] = new void*[MAX_PKT_BURST];
  burst->pkt_lens[0] = new uint32_t[MAX_PKT_BURST];
  burst->pkt_lens[1] = new uint32_t[MAX_PKT_BURST];
  memset(burst->pkt_lens[0], 0, MAX_PKT_BURST * sizeof(uint32_t));
  memset(burst->pkt_lens[1], 0, MAX_PKT_BURST * sizeof(uint32_t));

  m_burst_info.burst_id = burst_id;
  memcpy(get_burst_info(burst.get()), &m_burst_info, sizeof(m_burst_info));
  return burst;
}

/**
 * @brief Deletes a burst and frees its associated resources.
 *
 * @param burst A shared pointer to the burst to be deleted.
 */
void RxBurstsManager::delete_burst(std::shared_ptr<AdvNetBurstParams> burst) {
  if (burst == nullptr) {
    HOLOSCAN_LOG_ERROR("Invalid burst");
    return;
  }

  // Delete packet information if required
  if (m_send_packet_info && burst->pkt_extra_info != nullptr) {
    for (int i = 0; i < MAX_PKT_BURST; i++) {
      if (burst->pkt_extra_info[i] != nullptr) {
        delete reinterpret_cast<RmaxPacketExtendedInfo*>(burst->pkt_extra_info[i]);
        burst->pkt_extra_info[i] = nullptr;
      }
    }
    delete[] burst->pkt_extra_info;
    burst->pkt_extra_info = nullptr;
  }

  // Free the memory allocated for packets and their lengths
  delete[] burst->pkts[0];
  burst->pkts[0] = nullptr;
  delete[] burst->pkts[1];
  burst->pkts[1] = nullptr;
  delete[] burst->pkt_lens[0];
  burst->pkt_lens[0] = nullptr;
  delete[] burst->pkt_lens[1];
  burst->pkt_lens[1] = nullptr;
}

/**
 * @brief Constructor for the RxBurstsManager class.
 *
 * Initializes the chunk consumer with the specified parameters.
 *
 * @param send_packet_info Flag indicating if packet information should be sent.
 * @param port_id The port ID.
 * @param queue_id The queue ID.
 * @param burst_out_size The minimum output chunk size.
 * @param gpu_id The GPU ID.
 * @param rx_bursts_out_queue Shared pointer to the output queue for received bursts.
 */
RxBurstsManager::RxBurstsManager(bool send_packet_info, int port_id, int queue_id,
                                 uint16_t burst_out_size, int gpu_id,
                                 std::shared_ptr<IAnoBurstsCollection> rx_bursts_out_queue)
    : m_send_packet_info(send_packet_info),
      m_port_id(port_id),
      m_queue_id(queue_id),
      m_burst_out_size(burst_out_size),
      m_gpu_id(gpu_id),
      m_rx_bursts_out_queue(rx_bursts_out_queue) {
  const uint32_t burst_tag = (port_id << 16) | queue_id;
  m_gpu_direct = (m_gpu_id != INVALID_GPU_ID);

  // Initialize common burst info
  m_burst_info.tag = burst_tag;
  m_burst_info.burst_flags =
      (m_send_packet_info ? BurstFlags::INFO_PER_PACKET : BurstFlags::FLAGS_NONE);
  m_burst_info.burst_id = 0;
  m_burst_info.hds_on = false;
  m_burst_info.header_on_cpu = false;
  m_burst_info.payload_on_cpu = false;
  m_burst_info.header_stride_size = 0;
  m_burst_info.payload_stride_size = 0;
  m_burst_info.header_seg_idx = 0;
  m_burst_info.payload_seg_idx = 0;

  // Initialize the burst memory pool
  m_rx_bursts_mempool =
      std::make_unique<AnoBurstsMemoryPool>(DEFAULT_NUM_RX_BURSTS, *this, burst_tag);
  // Initialize the output queue if not provided
  if (!m_rx_bursts_out_queue) {
    m_rx_bursts_out_queue = std::make_shared<AnoBurstsQueue>();
    m_using_shared_out_queue = false;
  }

  // Adjust burst size to be within valid limits
  if (m_burst_out_size > MAX_PKT_BURST || m_burst_out_size == 0) m_burst_out_size = MAX_PKT_BURST;
}

/**
 * @brief Destructor for the RxBurstsManager class.
 *
 * Ensures that all bursts are properly returned to the memory pool.
 */
RxBurstsManager::~RxBurstsManager() {
  if (m_using_shared_out_queue) { return; }

  std::shared_ptr<AdvNetBurstParams> burst;
  // Get all bursts from the queue and return them to the memory pool
  while (m_rx_bursts_out_queue->available_bursts() > 0) {
    burst = m_rx_bursts_out_queue->dequeue_burst();
    if (burst == nullptr) break;
    m_rx_bursts_mempool->enqueue_burst(burst);
  }
}

};  // namespace holoscan::ops
