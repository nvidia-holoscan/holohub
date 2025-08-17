/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "advanced_network/common.h"
#include "advanced_network/kernels.h"
#include "kernels.cuh"
#include "holoscan/holoscan.hpp"
#include <queue>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/time.h>

using namespace holoscan::advanced_network;

namespace holoscan::ops {
/**
 * @brief: Holds the essential frame data.
 */
class FrameBuffer {
 public:
  /**
   * @brief: Constructor that allocates internal memory.
   *
   * @param [in] buffer_size: Size of the buffer to allocate.
   */
  explicit FrameBuffer(size_t buffer_size)
      : buffer_(new uint8_t[buffer_size]), buffer_size_(buffer_size) {}

  /**
   * @brief: Move constructor.
   */
  FrameBuffer(FrameBuffer&& other) noexcept
      : buffer_(std::move(other.buffer_)), buffer_size_(other.buffer_size_) {}

  /**
   * @brief: Move assignment operator.
   */
  FrameBuffer& operator=(FrameBuffer&& other) noexcept {
    if (this != &other) {
      buffer_ = std::move(other.buffer_);
      buffer_size_ = other.buffer_size_;
    }
    return *this;
  }

  // Delete copy constructor and assignment operator
  FrameBuffer(const FrameBuffer&) = delete;
  FrameBuffer& operator=(const FrameBuffer&) = delete;
  ~FrameBuffer() = default;

  /**
   * @brief: Return pointer to the buffer.
   *
   * @return: Pointer to the buffer.
   */
  uint8_t* get() const {
    return buffer_.get();
  }

  /**
   * @brief: Return size of the buffer.
   *
   * @return: Size of the buffer.
   */
  size_t get_size() const {
    return buffer_size_;
  }

 private:
  /* Smart pointer for the owned memory */
  std::unique_ptr<uint8_t[]> buffer_;
  /* Size of the buffer */
  size_t buffer_size_;
};

/**
 * @brief Thread-safe buffer for concurrent item production and consumption.
 *
 * This template class provides a thread-safe queue with blocking and non-blocking
 * operations for adding and retrieving items. It's designed for producer-consumer
 * scenarios where multiple threads may be accessing the buffer concurrently.
 *
 * @tparam T The type of items stored in the buffer
 */
template <typename T>
class ConcurrentItemBuffer {
 public:
  /**
   * @brief Constructor.
   *
   * @param [in] max_queue_size Maximum size of the internal queue (0 for unlimited).
   */
  explicit ConcurrentItemBuffer(size_t max_queue_size = 0)
      : max_queue_size_(max_queue_size), stop_(false) {}

  /**
   * @brief Destructor.
   */
  virtual ~ConcurrentItemBuffer() { stop(); }

  /**
   * @brief Get an item from the buffer, blocking if none is available.
   *
   * This method will block until an item is available or the buffer is stopped.
   *
   * @return Shared pointer to an item, or nullptr if the buffer is stopped.
   */
  std::shared_ptr<T> get_item_blocking() {
    std::unique_lock<std::mutex> lock(mutex_);
    // Wait until an item is available or stop is requested
    cv_.wait(lock, [this] { return !item_queue_.empty() || stop_; });

    if (stop_ && item_queue_.empty()) { return nullptr; }

    auto item = item_queue_.front();
    item_queue_.pop();

    // Notify any threads waiting to add items that space is now available
    if (max_queue_size_ > 0) { cv_.notify_one(); }

    return item;
  }

  /**
   * @brief Get an item from the buffer without blocking.
   *
   * @return Shared pointer to an item, or nullptr if no items are available.
   */
  std::shared_ptr<T> get_item_not_blocking() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (item_queue_.empty()) { return nullptr; }

    auto item = item_queue_.front();
    item_queue_.pop();

    // Notify any threads waiting to add items that space is now available
    if (max_queue_size_ > 0) { cv_.notify_one(); }

    return item;
  }

  /**
   * @brief Add an item to the buffer without blocking.
   *
   * @param [in] item Shared pointer to an item.
   * @return Status of the operation.
   */
  virtual Status add_item(std::shared_ptr<T> item) {
    if (!item) { return Status::NULL_PTR; }

    std::lock_guard<std::mutex> lock(mutex_);

    if (max_queue_size_ > 0 && item_queue_.size() >= max_queue_size_) {
      return Status::NO_SPACE_AVAILABLE;
    }

    item_queue_.push(std::move(item));
    cv_.notify_one();
    return Status::SUCCESS;
  }

  /**
   * @brief Add an item to the buffer, blocking if buffer is full.
   *
   * This method will block until there's space in the buffer or the buffer is stopped.
   *
   * @param [in] item Shared pointer to an item.
   * @return Status of the operation.
   */
  virtual Status add_item_blocking(std::shared_ptr<T> item) {
    if (!item) { return Status::NULL_PTR; }

    std::unique_lock<std::mutex> lock(mutex_);

    // Wait until there's space in the queue or stop is requested
    if (max_queue_size_ > 0) {
      cv_.wait(lock, [this] { return item_queue_.size() < max_queue_size_ || stop_; });
    }

    // If we're stopping and don't want to add more items
    if (stop_) { return Status::NOT_READY; }

    item_queue_.push(std::move(item));
    cv_.notify_one();  // Notify any waiting consumers
    return Status::SUCCESS;
  }

  /**
   * @brief Stop the buffer.
   *
   * This will unblock any threads waiting in get_item_blocking() or add_item_blocking().
   */
  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
    cv_.notify_all();
  }

  /**
   * @brief Return the number of items in the buffer.
   *
   * @return Number of items in the buffer.
   */
  size_t get_queue_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return item_queue_.size();
  }

  /**
   * @brief Check if more items can be added without blocking.
   *
   * @return True if items can be added, false if the buffer is full.
   */
  bool has_space() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_queue_size_ == 0 || item_queue_.size() < max_queue_size_;
  }

  /**
   * @brief Check if the buffer is empty.
   *
   * @return True if the buffer is empty, false otherwise.
   */
  bool is_empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return item_queue_.empty();
  }

  /**
   * @brief Get the maximum size of the buffer.
   *
   * @return Maximum size of the buffer (0 for unlimited).
   */
  size_t get_max_size() const { return max_queue_size_; }

 private:
  /* Queue of items */
  std::queue<std::shared_ptr<T>> item_queue_;
  /* Mutex for thread safety */
  mutable std::mutex mutex_;
  /* Condition variable for blocking operations */
  std::condition_variable cv_;
  /* Maximum size of the queue (0 for unlimited) */
  size_t max_queue_size_;
  /* Flag to indicate if the buffer is stopping */
  bool stop_;
};
/**
 * @brief: Reads frames from a file and streams them to a @ref BufferedMediaFrameProvider.
 *
 * This class reads media frames from a file, uses a @ref MediaFramePool for memory management,
 * and pushes the frames to a @ref BufferedMediaFrameProvider for consumption by other components.
 * It supports looping through the file when reaching the end and provides thread-safe
 * operations for starting and stopping the streaming process.
 */
class MediaFileFrameProvider {
 public:
  /**
   * @brief: Constructor.
   *
   * @param [in] file_path: Path to the media file.
   * @param [in] frame_size: Size of each frame in bytes.
   * @param [in] loop: Whether to loop through the file when reaching the end.
   * @param [in] sleep_duration_microseconds: Sleep duration in microseconds between reading frames.
   */
  MediaFileFrameProvider(const std::string& file_path, size_t frame_size, bool loop,
                         size_t sleep_duration_microseconds = SLEEP_DURATION_MICROSECONDS)
      : file_path_(file_path),
        frame_size_(frame_size),
        frame_buffer_queue_(FRAME_BUFFER_QUEUE_SIZE),
        loop_frames_(loop),
        sleep_duration_microseconds_(sleep_duration_microseconds) {}

  /**
   * @brief: Destructor.
   */
  virtual ~MediaFileFrameProvider() { stop(); }

  /**
   * @brief: Initialize the streaming process.
   *
   * @return: Status of the operation.
   */
  Status initialize() {
    if (initialized_) { return Status::SUCCESS; }

    input_file_.open(file_path_, std::ios::binary);
    if (!input_file_.is_open()) {
      HOLOSCAN_LOG_ERROR("Failed to open file: {}", file_path_);
      return Status::INVALID_PARAMETER;
    }

    initialized_ = true;
    return Status::SUCCESS;
  }

  /**
   * @brief: Stop the streaming process.
   */
  void stop() {
    stop_ = true;
    frame_buffer_queue_.stop();
  }
  /**
   * @brief: Returns number of available frames in the queue.
   *
   * This method returns the number of frames currently available in the queue.
   *
   * @return: Number of available frames.
   */
  size_t get_number_of_available_frames() { return frame_buffer_queue_.get_queue_size(); }

  /**
   * @brief: Get the next available frame.
   *
   * This method attempts to get a frame from the queue without blocking.
   * If no frames are available, it returns nullptr.
   *
   * @return: Shared pointer to a frame buffer, or nullptr if none available.
   */
  std::shared_ptr<FrameBuffer> get_next_frame() {
    return frame_buffer_queue_.get_item_not_blocking();
  }

  /**
   * @brief: Get the next frame, blocking if none available.
   *
   * This method blocks until a frame is available or the provider is stopped.
   *
   * @return: Shared pointer to a frame buffer, or nullptr if provider stopped.
   */
  std::shared_ptr<FrameBuffer> get_next_frame_blocking() {
    return frame_buffer_queue_.get_item_blocking();
  }

  /**
   * @brief: Call operator for running in a separate thread.
   */
  void operator()() {
    if (!initialized_) {
      HOLOSCAN_LOG_ERROR("MediaFileFrameProvider is not initialized");
      return;
    }
    if (!input_file_.is_open()) {
      HOLOSCAN_LOG_ERROR("File is not open: {}", file_path_);
      return;
    }

    while (!stop_) {
      auto frame_buffer = std::make_shared<FrameBuffer>(frame_size_);
      input_file_.read(reinterpret_cast<char*>(frame_buffer->get()), frame_size_);
      std::streamsize bytes_read = input_file_.gcount();

      if (bytes_read == 0) {
        if (input_file_.eof()) {
          if (!loop_frames_) { break; }
          // Loop the file, start reading from the beginning
          input_file_.clear();
          input_file_.seekg(0, std::ios::beg);
          continue;
        } else if (input_file_.fail()) {
          HOLOSCAN_LOG_ERROR("Error reading from file: {}", file_path_);
          break;
        }
      }

      // Handle partial frame read if needed
      if (bytes_read < static_cast<std::streamsize>(frame_size_)) {
        std::memset(frame_buffer->get() + bytes_read, 0, frame_size_ - bytes_read);
      }

      while (!stop_) {
        auto status = frame_buffer_queue_.add_item_blocking(frame_buffer);

        if (status == Status::SUCCESS) { break; }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_duration_microseconds_));
      }
    }

    input_file_.close();
    frame_buffer_queue_.stop();
    initialized_ = false;
  }

 private:
  static constexpr auto SLEEP_DURATION_MICROSECONDS = 10000;
  static constexpr auto FRAME_BUFFER_QUEUE_SIZE = 15;

  std::string file_path_;
  size_t frame_size_;
  ConcurrentItemBuffer<FrameBuffer> frame_buffer_queue_;
  bool loop_frames_;
  std::atomic<bool> stop_{false};
  std::ifstream input_file_;
  std::atomic<bool> initialized_{false};
  size_t sleep_duration_microseconds_;
};

void shutdown(std::string message) {
  advanced_network::shutdown();
  HOLOSCAN_LOG_ERROR("{}", message);
  exit(1);
}

class AdvNetworkingBenchRivermaxTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchRivermaxTxOp)

  AdvNetworkingBenchRivermaxTxOp() = default;

  ~AdvNetworkingBenchRivermaxTxOp() {
    // Stop the provider and join the provider thread
    if (frame_provider_) { frame_provider_->stop(); }
    if (provider_thread_.joinable()) { provider_thread_.join(); }
    HOLOSCAN_LOG_INFO("ANO benchmark Rivermax TX op shutting down");
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchRivermaxTxOp::initialize()");
    holoscan::Operator::initialize();

    port_id_ = get_port_id(interface_name_.get());
    if (port_id_ == -1) {
      HOLOSCAN_LOG_ERROR("Invalid TX port {} specified in the config", interface_name_.get());
      exit(1);
    }

    frame_size_ = calculate_frame_size(
        frame_width_.get(), frame_height_.get(), sample_format_.get(), bit_depth_.get());
    std::cout << "Frame size: " << frame_size_ << " bytes" << std::endl;

    frame_provider_ =
        std::make_unique<MediaFileFrameProvider>(file_path_.get(), frame_size_, loop_frames_);

    auto status = frame_provider_->initialize();
    if (status != Status::SUCCESS) {
      shutdown("Failed to initialize media file provider");
    }

    // Start the provider thread
    provider_thread_ = std::thread([this]() {
      (*frame_provider_)();  // Run the provider
    });

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchRivermaxTxOp::initialize() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.param<std::string>(interface_name_,
      "interface_name",
      "Name of NIC from advanced_network config",
      "Name of NIC from advanced_network config");
    spec.param<uint16_t>(queue_id_, "queue_id", "Queue ID", "Queue ID", default_queue_id);
    spec.param<uint32_t>(frame_width_, "frame_width", "Frame width", "Width of the frame", 1920);
    spec.param<uint32_t>(
        frame_height_, "frame_height", "Frame height", "Height of the frame", 1080);
    spec.param<uint32_t>(bit_depth_, "bit_depth", "Bit depth", "Number of bits per pixel", 8);
    spec.param<std::string>(
        sample_format_, "sample_format", "Sample Format", "Format of the frame", "RGB888");
    spec.param<std::string>(
        payload_memory_, "payload_memory", "Payload Memory", "Memory for the payload", "host");
    spec.param<std::string>(
        file_path_, "file_path", "File Path", "Path to the file", "data/frames.dat");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    static int not_available_count = 0;
    static int sent = 0;
    static int err = 0;
    if (!cur_msg_) {
      cur_msg_ = create_tx_burst_params();
      set_header(cur_msg_, port_id_, queue_id_.get(), 1, 1);
    }

    if (!is_tx_burst_available(cur_msg_)) {
      std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_WHEN_BURST_NOT_AVAILABLE_US));
      if (++not_available_count == DISPLAY_WARNING_AFTER_BURST_NOT_AVAILABLE) {
        HOLOSCAN_LOG_ERROR(
            "TX port {}, queue {}, burst not available too many times consecutively. "
            "Make sure memory region has enough buffers. Sent {} and error {}",
            port_id_,
            queue_id_.get(),
            sent,
            err);
        not_available_count = 0;
        err++;
      }
      return;
    }

    not_available_count = 0;
    Status ret;
    if ((ret = get_tx_packet_burst(cur_msg_)) != Status::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Error returned from get_tx_packet_burst: {}", static_cast<int>(ret));
      return;
    }

    auto frame = frame_provider_->get_next_frame();
    if (!frame) {
      HOLOSCAN_LOG_DEBUG("No frames available from provider");
      free_tx_burst(cur_msg_);
      return;
    }

    // Copy frame data into the burst
    if (payload_memory_.get() == "device") {
      cudaMemcpy(cur_msg_->pkts[0][0], frame->get(), frame->get_size(), cudaMemcpyDefault);
    } else {
      std::memcpy(cur_msg_->pkts[0][0], frame->get(), frame->get_size());
    }

    ret = send_tx_burst(cur_msg_);
    if (ret != Status::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Error returned from send_tx_burst: {}", static_cast<int>(ret));
      free_tx_burst(cur_msg_);
      err++;
    } else {
      sent++;
    }
    cur_msg_ = nullptr;
    HOLOSCAN_LOG_TRACE("AdvNetworkingBenchRivermaxTxOp::compute() {}:{} done. Emitted{}/Error{}",
                       port_id_,
                       queue_id_.get(),
                       sent,
                       err);
  }

  enum class VideoSampling { RGB, YCbCr_4_2_2, YCbCr_4_2_0, YCbCr_4_4_4, Unknown};
  enum class ColorBitDepth { _8, _10, _12, Unknown};
  using BytesPerPixelRatio = std::pair<uint32_t, uint32_t>;
  using ColorDepthPixelRatioMap =
      std::unordered_map<VideoSampling, std::unordered_map<ColorBitDepth, BytesPerPixelRatio>>;

  const ColorDepthPixelRatioMap COLOR_DEPTH_TO_PIXEL_RATIO = {
      {VideoSampling::RGB,
       {{ColorBitDepth::_8, {3, 1}}, {ColorBitDepth::_10, {15, 4}}, {ColorBitDepth::_12, {9, 2}}}},
      {VideoSampling::YCbCr_4_4_4,
       {{ColorBitDepth::_8, {3, 1}}, {ColorBitDepth::_10, {15, 4}}, {ColorBitDepth::_12, {9, 2}}}},
      {VideoSampling::YCbCr_4_2_2,
       {{ColorBitDepth::_8, {4, 2}}, {ColorBitDepth::_10, {5, 2}}, {ColorBitDepth::_12, {6, 2}}}},
      {VideoSampling::YCbCr_4_2_0,
       {{ColorBitDepth::_8, {6, 4}}, {ColorBitDepth::_10, {15, 8}}, {ColorBitDepth::_12, {9, 4}}}}};

  /**
   * @brief: Returns the corresponding VideoSampling enum value for the given video sampling format.
   *
   * @param format a string representing the video sampling format
   *
   * @return the corresponding VideoSampling enum value
   *
   * @note exits the application if the video sampling format is not supported
   */
  VideoSampling get_video_sampling_format(const std::string& format) {
    if (format == "RGB888") return VideoSampling::RGB;
    if (format == "YUV422") return VideoSampling::YCbCr_4_2_2;
    if (format == "YUV420") return VideoSampling::YCbCr_4_2_0;
    if (format == "YUV442") return VideoSampling::YCbCr_4_4_4;
    return VideoSampling::Unknown;
  }

  /**
   * @brief Returns the corresponding ColorBitDepth enum value for the given bit depth.
   *
   * @param bit_depth the bit depth to map
   *
   * @return the corresponding ColorBitDepth enum value
   *
   * @note exits the application if the bit depth is not supported
   */
  ColorBitDepth get_color_bit_depth(int bit_depth) {
    switch (bit_depth) {
      case 8:
        return ColorBitDepth::_8;
      case 10:
        return ColorBitDepth::_10;
      case 12:
        return ColorBitDepth::_12;
      default:
        return ColorBitDepth::Unknown;
    }
  }

  /**
   * @brief Calculates the size of a frame based on its width, height, format, and bit depth.
   *
   * @param width The width of the frame.
   * @param height The height of the frame.
   * @param format_str The format of the frame.
   * @param bit_depth_int The bit depth of the frame.
   *
   * @return The size of the frame in bytes.
   *
   * @note exits the application if the format or bit depth is unsupported.
   */
  size_t calculate_frame_size(uint32_t width, uint32_t height, const std::string& format_str,
                              uint32_t bit_depth_int) {
    VideoSampling sampling_format = get_video_sampling_format(format_str);
    ColorBitDepth bit_depth = get_color_bit_depth(bit_depth_int);

    auto format_it = COLOR_DEPTH_TO_PIXEL_RATIO.find(sampling_format);
    if (format_it == COLOR_DEPTH_TO_PIXEL_RATIO.end()) {
      shutdown("Unsupported sampling format");
    }

    auto depth_it = format_it->second.find(bit_depth);
    if (depth_it == format_it->second.end()) {
      shutdown("Unsupported bit depth");
    }

    float bytes_per_pixel = static_cast<float>(depth_it->second.first) / depth_it->second.second;

    return static_cast<size_t>(width * height * bytes_per_pixel);
  }

 private:
  static constexpr uint16_t default_queue_id = 0;
  static constexpr auto SLEEP_WHEN_BURST_NOT_AVAILABLE_US = 1000;
  static constexpr auto DISPLAY_WARNING_AFTER_BURST_NOT_AVAILABLE = 10000;
  int port_id_ = -1;
  Parameter<std::string> interface_name_;
  Parameter<uint16_t> queue_id_;
  size_t frame_size_;
  Parameter<uint32_t> frame_width_;
  Parameter<uint32_t> frame_height_;
  Parameter<std::string> sample_format_;
  Parameter<uint32_t> bit_depth_;
  Parameter<std::string> payload_memory_;
  Parameter<std::string> file_path_;
  bool loop_frames_ = true;
  std::unique_ptr<MediaFileFrameProvider> frame_provider_;
  std::thread provider_thread_;
  BurstParams* cur_msg_ = nullptr;
};

}  // namespace holoscan::ops
