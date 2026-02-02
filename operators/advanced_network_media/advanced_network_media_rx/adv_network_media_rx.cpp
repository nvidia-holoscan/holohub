/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda_runtime.h>
#include <chrono>
#include <memory>
#include <string>
#include <deque>
#include <stdexcept>

#include "adv_network_media_rx.h"
#include "advanced_network/common.h"
#include "../common/adv_network_media_common.h"
#include "media_frame_assembler.h"
#include "network_burst_processor.h"
#include <holoscan/utils/cuda_stream_handler.hpp>
#include <holoscan/logger/logger.hpp>
#include "../common/frame_buffer.h"
#include "../common/video_parameters.h"
#include "advanced_network/managers/rivermax/rivermax_ano_data_types.h"

namespace holoscan::ops {

namespace ano = holoscan::advanced_network;
using holoscan::advanced_network::AnoBurstExtendedInfo;
using holoscan::advanced_network::BurstParams;
using holoscan::advanced_network::Status;

constexpr size_t FRAMES_IN_POOL = 250;
constexpr size_t PACKETS_DISPLAY_INTERVAL = 1000000;  // 1e6 packets

#if ENABLE_STATISTICS_LOGGING
// Statistics timing constants
constexpr size_t STATS_REPORT_INTERVAL_MS = 30000;  // Report every 30 seconds
constexpr size_t STATS_WINDOW_SIZE_MS = 30000;      // Calculate rates over 30 seconds

// Constraint: Window must be at least as large as report interval to have enough samples
static_assert(STATS_WINDOW_SIZE_MS >= STATS_REPORT_INTERVAL_MS,
              "STATS_WINDOW_SIZE_MS must be >= STATS_REPORT_INTERVAL_MS to ensure "
              "sufficient samples for rate calculation");

// Structure to hold timestamped statistics samples
struct StatsSample {
  std::chrono::steady_clock::time_point timestamp;
  size_t packets_received;
  size_t bursts_processed;
  size_t frames_emitted;
};
#endif

// Enumeration for output format types
enum class OutputFormatType { VIDEO_BUFFER, TENSOR };

/**
 * @brief Frame completion handler for the RX operator
 */
class RxOperatorFrameCompletionHandler : public IFrameCompletionHandler {
 public:
  explicit RxOperatorFrameCompletionHandler(class AdvNetworkMediaRxOpImpl* impl) : impl_(impl) {}

  void on_frame_completed(std::shared_ptr<FrameBufferBase> frame) override;
  void on_frame_error(const std::string& error_message) override;

 private:
  class AdvNetworkMediaRxOpImpl* impl_;
};

/**
 * @class AdvNetworkMediaRxOpImpl
 * @brief Implementation class for the AdvNetworkMediaRxOp operator.
 *
 * Handles high-level network management, frame pool management, and
 * coordinates with MediaFrameAssembler for packet-level operations.
 */
class AdvNetworkMediaRxOpImpl : public IFrameProvider {
 public:
  /**
   * @brief Constructs an implementation for the given operator.
   *
   * @param parent Reference to the parent operator.
   */
  explicit AdvNetworkMediaRxOpImpl(AdvNetworkMediaRxOp& parent) : parent_(parent) {}

  /**
   * @brief Initializes the implementation.
   *
   * Sets up the network port, allocates frame buffers, and prepares
   * for media reception.
   */
  void initialize() {
    ANM_LOG_INFO("AdvNetworkMediaRxOp::initialize()");

    // Initialize timing for statistics
#if ENABLE_STATISTICS_LOGGING
    start_time_ = std::chrono::steady_clock::now();
    last_stats_report_ = start_time_;

    // Initialize rolling window with first sample
    StatsSample initial_sample{start_time_, 0, 0, 0};
    stats_samples_.push_back(initial_sample);
#endif

    port_id_ = ano::get_port_id(parent_.interface_name_.get());
    if (port_id_ == -1) {
      std::string error_message = "Invalid RX port interface name '" +
                                  std::string(parent_.interface_name_.get()) +
                                  "' specified in the config";
      ANM_LOG_ERROR("Invalid RX port {} specified in the config", parent_.interface_name_.get());
      throw std::runtime_error(error_message);
    } else {
      ANM_CONFIG_LOG("RX port {} found", port_id_);
    }

    // Convert params to video parameters
    auto video_sampling = get_video_sampling_format(parent_.video_format_.get());
    auto color_bit_depth = get_color_bit_depth(parent_.bit_depth_.get());

    // Get video format and calculate frame size
    video_format_ = get_expected_gxf_video_format(video_sampling, color_bit_depth);
    frame_size_ = calculate_frame_size(
        parent_.frame_width_.get(), parent_.frame_height_.get(), video_sampling, color_bit_depth);

    // Determine output format type
    const auto& output_format_str = parent_.output_format_.get();
    if (output_format_str == "tensor") {
      output_format_ = OutputFormatType::TENSOR;
      ANM_CONFIG_LOG("Using Tensor output format");
    } else {
      output_format_ = OutputFormatType::VIDEO_BUFFER;
      ANM_CONFIG_LOG("Using VideoBuffer output format");
    }

    // Determine memory location type for output frames
    const auto& memory_location_str = parent_.memory_location_.get();
    if (memory_location_str == "host") {
      storage_type_ = nvidia::gxf::MemoryStorageType::kHost;
      ANM_CONFIG_LOG("Using Host memory location for output frames");
    } else {
      storage_type_ = nvidia::gxf::MemoryStorageType::kDevice;
      ANM_CONFIG_LOG("Using Device memory location for output frames");
    }

    // Create pool of allocated frame buffers
    create_frame_pool();

    // Create media frame assembler and network burst processor
    create_media_frame_assembler();

    // Create network burst processor
    burst_processor_ = std::make_unique<NetworkBurstProcessor>(assembler_);
  }

  /**
   * @brief Creates a pool of frame buffers for receiving frames.
   */
  void create_frame_pool() {
    frames_pool_.clear();

    // Get the appropriate channel count based on video format
    uint32_t channels = get_channel_count_for_format(video_format_);

    for (size_t i = 0; i < FRAMES_IN_POOL; ++i) {
      void* data = nullptr;

      // Allocate memory based on storage type
      if (storage_type_ == nvidia::gxf::MemoryStorageType::kHost) {
        CUDA_TRY(cudaMallocHost(&data, frame_size_));
      } else {
        CUDA_TRY(cudaMalloc(&data, frame_size_));
      }

      // Create appropriate frame buffer type
      if (output_format_ == OutputFormatType::TENSOR) {
        frames_pool_.push_back(std::make_shared<AllocatedTensorFrameBuffer>(
            data,
            frame_size_,
            parent_.frame_width_.get(),
            parent_.frame_height_.get(),
            channels,  // Use format-specific channel count
            video_format_,
            storage_type_));
      } else {
        frames_pool_.push_back(
            std::make_shared<AllocatedVideoBufferFrameBuffer>(data,
                                                              frame_size_,
                                                              parent_.frame_width_.get(),
                                                              parent_.frame_height_.get(),
                                                              video_format_,
                                                              storage_type_));
      }
    }
  }

  /**
   * @brief Creates the media frame assembler with minimal configuration
   * @note Full configuration will be done when first burst arrives
   */
  void create_media_frame_assembler() {
    // Create minimal assembler configuration (will be completed from burst data)
    auto config = AssemblerConfiguration{};

    // Set operator-known parameters only
    config.source_memory_type =
        nvidia::gxf::MemoryStorageType::kHost;  // Will be updated from burst
    config.destination_memory_type = storage_type_;
    config.enable_memory_copy_strategy_detection = true;
    config.force_contiguous_memory_copy_strategy = false;

    // Create frame provider (this class implements IFrameProvider)
    auto frame_provider = std::shared_ptr<IFrameProvider>(this, [](IFrameProvider*) {});

    // Create frame assembler with minimal config
    assembler_ = std::make_shared<MediaFrameAssembler>(frame_provider, config);

    // Create completion handler
    completion_handler_ = std::make_shared<RxOperatorFrameCompletionHandler>(this);
    assembler_->set_completion_handler(completion_handler_);

    ANM_CONFIG_LOG("Media frame assembler created with minimal configuration");
  }

  /**
   * @brief Clean up resources properly when the operator is destroyed.
   */
  void cleanup() {
    // Free all allocated frames
    for (auto& frame : frames_pool_) {
      if (storage_type_ == nvidia::gxf::MemoryStorageType::kHost) {
        CUDA_TRY(cudaFreeHost(frame->get()));
      } else {
        CUDA_TRY(cudaFree(frame->get()));
      }
    }
    frames_pool_.clear();

    // Free any frames in the ready queue
    for (auto& frame : ready_frames_) {
      if (storage_type_ == nvidia::gxf::MemoryStorageType::kHost) {
        CUDA_TRY(cudaFreeHost(frame->get()));
      } else {
        CUDA_TRY(cudaFree(frame->get()));
      }
    }
    ready_frames_.clear();

    // Free all in-flight network bursts awaiting cleanup
    while (!bursts_awaiting_cleanup_.empty()) {
      auto burst_to_free = bursts_awaiting_cleanup_.front();
      ano::free_all_packets_and_burst_rx(burst_to_free);
      bursts_awaiting_cleanup_.pop_front();
    }
    bursts_awaiting_cleanup_.clear();
  }

  /**
   * @brief Processes a received burst of packets and generates video frames.
   *
   * @param op_input The operator input context.
   * @param op_output The operator output context.
   * @param context The execution context.
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    compute_calls_++;

    BurstParams* burst;
    auto status = ano::get_rx_burst(&burst, port_id_, parent_.queue_id_.get());
    if (status != Status::SUCCESS)
      return;

    const auto& packets_received = burst->hdr.hdr.num_pkts;
    total_packets_received_ += packets_received;
    total_bursts_processed_++;

    if (packets_received == 0) {
      ano::free_all_packets_and_burst_rx(burst);
      return;
    }

    // Report periodic comprehensive statistics
    report_periodic_statistics();

    ANM_FRAME_TRACE("Processing burst: port={}, queue={}, packets={}, burst_ptr={}",
                    port_id_,
                    parent_.queue_id_.get(),
                    packets_received,
                    static_cast<void*>(burst));

    append_to_frame(burst);

    if (ready_frames_.empty())
      return;

    size_t total_frames = ready_frames_.size();

    if (total_frames > 1) {
      size_t frames_to_drop = total_frames - 1;
      total_frames_dropped_ += frames_to_drop;
      ANM_LOG_WARN(
          "Multiple frames ready ({}), dropping {} earlier frames to prevent pipeline issues",
          total_frames,
          frames_to_drop);
    }

    ANM_FRAME_TRACE("Ready frames count: {}, processing frame emission", total_frames);

    // Pop all frames but keep only the last one
    std::shared_ptr<FrameBufferBase> last_frame = nullptr;
    while (auto frame = pop_ready_frame()) {
      if (last_frame) {
        // Return the previous frame back to pool (dropping it)
        frames_pool_.push_back(last_frame);
        ANM_FRAME_TRACE("Dropped frame returned to pool: size={}, ptr={}",
                        last_frame->get_size(),
                        static_cast<void*>(last_frame->get()));
      }
      last_frame = frame;
    }

    // Emit only the last (most recent) frame
    if (last_frame) {
      total_frames_emitted_++;
      ANM_FRAME_TRACE("Emitting latest frame: {} bytes", last_frame->get_size());
      ANM_FRAME_TRACE("Frame emission details: size={}, ptr={}, memory_location={}",
                      last_frame->get_size(),
                      static_cast<void*>(last_frame->get()),
                      static_cast<int>(last_frame->get_memory_location()));
      auto result = create_frame_entity(last_frame, context);
      op_output.emit(result);
    }
  }

  /**
   * @brief Appends packet data from a burst to the current frame being constructed.
   *
   * @param burst The burst containing packets to process.
   */
  void append_to_frame(BurstParams* burst) {
    // Configure assembler on first burst
    if (!assembler_configured_) {
      configure_assembler_from_burst(burst);
      assembler_configured_ = true;
    }

    size_t ready_frames_before = ready_frames_.size();

    ANM_FRAME_TRACE("Processing burst: ready_frames_before={}, queue_size={}, burst_packets={}",
                    ready_frames_before,
                    bursts_awaiting_cleanup_.size(),
                    burst->hdr.hdr.num_pkts);

    burst_processor_->process_burst(burst);

    size_t ready_frames_after = ready_frames_.size();

    ANM_FRAME_TRACE("Burst processed: ready_frames_after={}, frames_completed={}",
                    ready_frames_after,
                    ready_frames_after - ready_frames_before);

    // If new frames were completed, free all accumulated bursts
    if (ready_frames_after > ready_frames_before) {
      size_t frames_completed = ready_frames_after - ready_frames_before;
      ANM_FRAME_TRACE("{} frame(s) completed, freeing {} accumulated bursts",
                      frames_completed,
                      bursts_awaiting_cleanup_.size());
      ANM_FRAME_TRACE("Freeing accumulated bursts: count={}", bursts_awaiting_cleanup_.size());
      while (!bursts_awaiting_cleanup_.empty()) {
        auto burst_to_free = bursts_awaiting_cleanup_.front();
        ANM_FRAME_TRACE("Freeing burst: ptr={}, packets={}",
                        static_cast<void*>(burst_to_free),
                        burst_to_free->hdr.hdr.num_pkts);
        ano::free_all_packets_and_burst_rx(burst_to_free);
        bursts_awaiting_cleanup_.pop_front();
      }
    }

    // Add current burst to the queue after freeing previous bursts
    bursts_awaiting_cleanup_.push_back(burst);

    ANM_FRAME_TRACE("Final queue_size={}, burst_ptr={}",
                    bursts_awaiting_cleanup_.size(),
                    static_cast<void*>(burst));
  }

  /**
   * @brief Retrieves a ready frame from the queue if available.
   *
   * @return Shared pointer to a ready frame, or nullptr if none is available.
   */
  std::shared_ptr<FrameBufferBase> pop_ready_frame() {
    if (ready_frames_.empty()) {
      return nullptr;
    }

    auto frame = ready_frames_.front();
    ready_frames_.pop_front();
    return frame;
  }

  /**
   * @brief Creates a GXF entity containing the frame for output.
   *
   * @param frame The frame to wrap.
   * @param context The execution context.
   * @return The GXF entity ready for emission.
   */
  holoscan::gxf::Entity create_frame_entity(std::shared_ptr<FrameBufferBase> frame,
                                            ExecutionContext& context) {
    // Create lambda to return frame to pool when done
    auto release_func = [this, frame](void*) -> nvidia::gxf::Expected<void> {
      frames_pool_.push_back(frame);
      return {};
    };

    if (output_format_ == OutputFormatType::TENSOR) {
      // Cast to AllocatedTensorFrameBuffer and wrap
      auto tensor_frame = std::static_pointer_cast<AllocatedTensorFrameBuffer>(frame);
      auto entity = tensor_frame->wrap_in_entity(context.context(), release_func);
      return holoscan::gxf::Entity(std::move(entity));
    } else {
      // Cast to AllocatedVideoBufferFrameBuffer and wrap
      auto video_frame = std::static_pointer_cast<AllocatedVideoBufferFrameBuffer>(frame);
      auto entity = video_frame->wrap_in_entity(context.context(), release_func);
      return holoscan::gxf::Entity(std::move(entity));
    }
  }

  // Frame management methods (used internally)
  std::shared_ptr<FrameBufferBase> get_allocated_frame() {
    if (frames_pool_.empty()) {
      throw std::runtime_error("Running out of resources, frames pool is empty");
    }
    auto frame = frames_pool_.front();
    frames_pool_.pop_front();
    return frame;
  }

  void on_new_frame(std::shared_ptr<FrameBufferBase> frame) {
    ready_frames_.push_back(frame);
    ANM_FRAME_TRACE("New frame ready: {}", frame->get_size());
  }

  std::shared_ptr<FrameBufferBase> get_new_frame() override {
    return get_allocated_frame();
  }

  size_t get_frame_size() const override {
    return frame_size_;
  }

  bool has_available_frames() const override {
    return !frames_pool_.empty();
  }

  void return_frame_to_pool(std::shared_ptr<FrameBufferBase> frame) override {
    if (frame) {
      frames_pool_.push_back(frame);
      ANM_FRAME_TRACE("Frame returned to pool: pool_size={}, frame_ptr={}",
                      frames_pool_.size(),
                      static_cast<void*>(frame->get()));
    }
  }

 private:
  /**
   * @brief Configure assembler with burst parameters and validate against operator parameters
   * @param burst The burst containing configuration info
   */
  void configure_assembler_from_burst(BurstParams* burst) {
    // Access burst extended info from custom_burst_data
    const auto* burst_info =
        reinterpret_cast<const AnoBurstExtendedInfo*>(&(burst->hdr.custom_burst_data));

    // Validate operator parameters against burst data
    validate_configuration_consistency(burst_info);

    // Configure assembler with burst parameters
    assembler_->configure_burst_parameters(
        burst_info->header_stride_size, burst_info->payload_stride_size, burst_info->hds_on);

    // Configure memory types based on burst info
    nvidia::gxf::MemoryStorageType src_type = burst_info->payload_on_cpu
                                                  ? nvidia::gxf::MemoryStorageType::kHost
                                                  : nvidia::gxf::MemoryStorageType::kDevice;

    assembler_->configure_memory_types(src_type, storage_type_);

    ANM_CONFIG_LOG(
        "Assembler configured from burst: header_stride={}, payload_stride={}, "
        "hds_on={}, payload_on_cpu={}, src_memory={}, dst_memory={}",
        burst_info->header_stride_size,
        burst_info->payload_stride_size,
        burst_info->hds_on,
        burst_info->payload_on_cpu,
        static_cast<int>(src_type),
        static_cast<int>(storage_type_));
  }

  /**
   * @brief Validate consistency between operator parameters and burst configuration
   * @param burst_info The burst configuration data
   */
  void validate_configuration_consistency(const AnoBurstExtendedInfo* burst_info) {
    // Validate HDS configuration
    bool operator_hds = parent_.hds_.get();
    bool burst_hds = burst_info->hds_on;

    if (operator_hds != burst_hds) {
      ANM_LOG_WARN(
          "HDS configuration mismatch: operator parameter={}, burst data={} - using burst data as "
          "authoritative",
          operator_hds,
          burst_hds);
    }

    ANM_CONFIG_LOG("Configuration validation completed: operator_hds={}, burst_hds={}",
                   operator_hds,
                   burst_hds);
  }

  /**
   * @brief Report periodic statistics for monitoring
   */
  void report_periodic_statistics() {
#if ENABLE_STATISTICS_LOGGING
    auto now = std::chrono::steady_clock::now();
    auto time_since_last_report =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_report_).count();

    if (time_since_last_report >= STATS_REPORT_INTERVAL_MS) {
      // Add current sample to rolling window
      StatsSample current_sample{now,
                                 total_packets_received_,
                                 total_bursts_processed_,
                                 total_frames_emitted_};
      stats_samples_.push_back(current_sample);

      // Remove samples outside the window, but always keep at least 2 samples for rate calculation
      auto window_start_time = now - std::chrono::milliseconds(STATS_WINDOW_SIZE_MS);
      while (stats_samples_.size() > 2 &&
             stats_samples_.front().timestamp < window_start_time) {
        stats_samples_.pop_front();
      }

      // Calculate rates based on rolling window
      double window_packets_per_sec = 0.0;
      double window_frames_per_sec = 0.0;
      double window_bursts_per_sec = 0.0;
      double actual_window_duration_sec = 0.0;

      if (stats_samples_.size() >= 2) {
        const auto& oldest = stats_samples_.front();
        const auto& newest = stats_samples_.back();

        auto window_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            newest.timestamp - oldest.timestamp).count();
        actual_window_duration_sec = window_duration_ms / 1000.0;

        if (actual_window_duration_sec > 0.0) {
          size_t window_packets = newest.packets_received - oldest.packets_received;
          size_t window_bursts = newest.bursts_processed - oldest.bursts_processed;
          size_t window_frames = newest.frames_emitted - oldest.frames_emitted;

          window_packets_per_sec = window_packets / actual_window_duration_sec;
          window_frames_per_sec = window_frames / actual_window_duration_sec;
          window_bursts_per_sec = window_bursts / actual_window_duration_sec;
        }
      }

      // Calculate lifetime averages
      auto total_runtime =
          std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
      double lifetime_packets_per_sec =
          total_runtime > 0 ? static_cast<double>(total_packets_received_) / total_runtime : 0.0;
      double lifetime_frames_per_sec =
          total_runtime > 0 ? static_cast<double>(total_frames_emitted_) / total_runtime : 0.0;
      double lifetime_bursts_per_sec =
          total_runtime > 0 ? static_cast<double>(total_bursts_processed_) / total_runtime : 0.0;

      ANM_STATS_LOG("AdvNetworkMediaRx Statistics Report:");
      ANM_STATS_LOG("  Runtime: {} seconds", total_runtime);
      ANM_STATS_LOG("  Total packets received: {}", total_packets_received_);
      ANM_STATS_LOG("  Total bursts processed: {}", total_bursts_processed_);
      ANM_STATS_LOG("  Total frames emitted: {}", total_frames_emitted_);
      ANM_STATS_LOG("  Total frames dropped: {}", total_frames_dropped_);
      ANM_STATS_LOG("  Compute calls: {}", compute_calls_);

      // Report current rates with actual measurement window
      if (actual_window_duration_sec > 0.0) {
        ANM_STATS_LOG(
            "  Current rates (over {:.1f}s): {:.2f} packets/sec, {:.2f} frames/sec, "
            "{:.2f} bursts/sec",
            actual_window_duration_sec,
            window_packets_per_sec,
            window_frames_per_sec,
            window_bursts_per_sec);
      } else {
        ANM_STATS_LOG("  Current rates: N/A (insufficient samples, need {} more seconds)",
                      STATS_REPORT_INTERVAL_MS / 1000);
      }

      ANM_STATS_LOG(
          "  Lifetime avg rates: {:.2f} packets/sec, {:.2f} frames/sec, "
          "{:.2f} bursts/sec",
          lifetime_packets_per_sec,
          lifetime_frames_per_sec,
          lifetime_bursts_per_sec);
      ANM_STATS_LOG("  Ready frames queue: {}, Burst cleanup queue: {}",
                    ready_frames_.size(),
                    bursts_awaiting_cleanup_.size());
      ANM_STATS_LOG("  Frame pool available: {}", frames_pool_.size());

      // Report assembler statistics if available
      if (assembler_) {
        auto assembler_stats = assembler_->get_statistics();
        ANM_STATS_LOG("  Frame Assembler - Packets: {}, Frames completed: {}, Errors recovered: {}",
                      assembler_stats.packets_processed,
                      assembler_stats.frames_completed,
                      assembler_stats.errors_recovered);
        ANM_STATS_LOG("  Frame Assembler - Current state: {}, Strategy: {}",
                      assembler_stats.current_frame_state,
                      assembler_stats.current_strategy);
      }

      last_stats_report_ = now;
    }
#endif
  }

 private:
  AdvNetworkMediaRxOp& parent_;
  int port_id_;

  // Frame assembly components
  std::shared_ptr<MediaFrameAssembler> assembler_;
  std::shared_ptr<RxOperatorFrameCompletionHandler> completion_handler_;
  std::unique_ptr<NetworkBurstProcessor> burst_processor_;

  // Frame management
  std::deque<std::shared_ptr<FrameBufferBase>> frames_pool_;
  std::deque<std::shared_ptr<FrameBufferBase>> ready_frames_;
  std::deque<BurstParams*> bursts_awaiting_cleanup_;

  // Enhanced statistics and configuration
  size_t total_packets_received_ = 0;
  size_t total_bursts_processed_ = 0;
  size_t total_frames_emitted_ = 0;
  size_t total_frames_dropped_ = 0;
  size_t compute_calls_ = 0;
#if ENABLE_STATISTICS_LOGGING
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_stats_report_;
  std::deque<StatsSample> stats_samples_;  // Rolling window of statistics samples
#endif

  nvidia::gxf::VideoFormat video_format_;
  size_t frame_size_;
  OutputFormatType output_format_{OutputFormatType::VIDEO_BUFFER};
  nvidia::gxf::MemoryStorageType storage_type_{nvidia::gxf::MemoryStorageType::kDevice};
  bool assembler_configured_ = false;  ///< Whether assembler has been configured from burst data
};

// ========================================================================================
// RxOperatorFrameCompletionHandler Implementation
// ========================================================================================

void RxOperatorFrameCompletionHandler::on_frame_completed(std::shared_ptr<FrameBufferBase> frame) {
  if (!impl_ || !frame)
    return;

  // Add completed frame to ready queue (same as old on_new_frame)
  impl_->on_new_frame(frame);

  ANM_FRAME_TRACE("Frame assembly completed: {} bytes", frame->get_size());
}

void RxOperatorFrameCompletionHandler::on_frame_error(const std::string& error_message) {
  if (!impl_)
    return;

  ANM_LOG_ERROR("Frame assembly error: {}", error_message);
  // Could add error statistics or recovery logic here
}

// ========================================================================================
// AdvNetworkMediaRxOp Implementation
// ========================================================================================

AdvNetworkMediaRxOp::AdvNetworkMediaRxOp() : pimpl_(nullptr) {}

AdvNetworkMediaRxOp::~AdvNetworkMediaRxOp() {
  if (pimpl_) {
    pimpl_->cleanup();  // Clean up allocated resources
    delete pimpl_;
    pimpl_ = nullptr;
  }
}

void AdvNetworkMediaRxOp::setup(OperatorSpec& spec) {
  ANM_LOG_INFO("AdvNetworkMediaRxOp::setup() - Configuring operator parameters");

  spec.output<nvidia::gxf::Entity>("out_video_buffer");
  spec.param<std::string>(interface_name_,
                          "interface_name",
                          "Name of NIC from advanced_network config",
                          "Name of NIC from advanced_network config");
  spec.param<uint16_t>(queue_id_, "queue_id", "Queue ID", "Queue ID", default_queue_id);
  spec.param<uint32_t>(frame_width_, "frame_width", "Frame width", "Width of the frame", 1920);
  spec.param<uint32_t>(frame_height_, "frame_height", "Frame height", "Height of the frame", 1080);
  spec.param<uint32_t>(bit_depth_, "bit_depth", "Bit depth", "Number of bits per pixel", 8);
  spec.param(
      video_format_, "video_format", "Video Format", "Video sample format", std::string("RGB888"));
  spec.param<bool>(hds_,
                   "hds",
                   "Header Data Split",
                   "The packets received split Data in the GPU and Headers in the CPU");
  spec.param(output_format_,
             "output_format",
             "Output Format",
             "Output format type ('video_buffer' or 'tensor')",
             std::string("video_buffer"));
  spec.param(memory_location_,
             "memory_location",
             "Memory Location for Output Frames",
             "Memory location for output frames ('host' or 'devices')",
             std::string("device"));

  ANM_CONFIG_LOG("AdvNetworkMediaRxOp setup completed - parameters registered");
}

void AdvNetworkMediaRxOp::initialize() {
  ANM_LOG_INFO("AdvNetworkMediaRxOp::initialize() - Starting operator initialization");
  holoscan::Operator::initialize();

  ANM_CONFIG_LOG("Creating implementation instance for AdvNetworkMediaRxOp");
  if (!pimpl_) {
    pimpl_ = std::make_unique<AdvNetworkMediaRxOpImpl>(*this);
  }

  ANM_CONFIG_LOG(
      "Initializing implementation with parameters: interface={}, queue_id={}, frame_size={}x{}, "
      "format={}, hds={}, output_format={}, memory={}",
      interface_name_.get(),
      queue_id_.get(),
      frame_width_.get(),
      frame_height_.get(),
      video_format_.get(),
      hds_.get(),
      output_format_.get(),
      memory_location_.get());

  pimpl_->initialize();

  ANM_LOG_INFO("AdvNetworkMediaRxOp initialization completed successfully");
}

void AdvNetworkMediaRxOp::compute(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext& context) {
  ANM_FRAME_TRACE("AdvNetworkMediaRxOp::compute() - Processing frame");

#if ENABLE_PERFORMANCE_LOGGING
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  try {
    pimpl_->compute(op_input, op_output, context);

#if ENABLE_PERFORMANCE_LOGGING
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    ANM_PERF_LOG("AdvNetworkMediaRxOp::compute() completed in {} microseconds", duration.count());
#endif
  } catch (const std::exception& e) {
    ANM_LOG_ERROR("AdvNetworkMediaRxOp::compute() failed with exception: {}", e.what());
    throw;
  } catch (...) {
    ANM_LOG_ERROR("AdvNetworkMediaRxOp::compute() failed with unknown exception");
    throw;
  }
}

};  // namespace holoscan::ops
