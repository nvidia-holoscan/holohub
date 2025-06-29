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
#include <cuda_runtime.h>

#include "adv_network_media_rx.h"
#include "advanced_network/common.h"
#include "../common/adv_network_media_common.h"
#include "packets_to_frames_converter.h"
#include "burst_processor.h"
#include <holoscan/utils/cuda_stream_handler.hpp>
#include "../common/frame_buffer.h"
#include "../common/video_parameters.h"
#include "advanced_network/managers/rivermax/rivermax_ano_data_types.h"

using namespace holoscan::advanced_network;

namespace holoscan::ops {

constexpr size_t FRAMES_IN_POOL = 50;
constexpr size_t PACKETS_DISPLAY_INTERVAL = 1000000;  // 1e6 packets

// Enumeration for output format types
enum class OutputFormatType { VIDEO_BUFFER, TENSOR };

/**
 * @class AdvNetworkMediaRxOpImpl
 * @brief Implementation class for the AdvNetworkMediaRxOp operator.
 *
 * Handles high-level network management, frame pool management, and
 * coordinates with BurstProcessor for packet-level operations.
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
    HOLOSCAN_LOG_INFO("AdvNetworkMediaRxOp::initialize()");
    port_id_ = get_port_id(parent_.interface_name_.get());
    if (port_id_ == -1) {
      HOLOSCAN_LOG_ERROR("Invalid RX port {} specified in the config",
                         parent_.interface_name_.get());
      exit(1);
    } else {
      HOLOSCAN_LOG_INFO("RX port {} found", port_id_);
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
      HOLOSCAN_LOG_INFO("Using Tensor output format");
    } else {
      output_format_ = OutputFormatType::VIDEO_BUFFER;
      HOLOSCAN_LOG_INFO("Using VideoBuffer output format");
    }

    // Determine memory location type for output frames
    const auto& memory_location_str = parent_.memory_location_.get();
    if (memory_location_str == "host") {
      storage_type_ = nvidia::gxf::MemoryStorageType::kHost;
      HOLOSCAN_LOG_INFO("Using Host memory location for output frames");
    } else {
      storage_type_ = nvidia::gxf::MemoryStorageType::kDevice;
      HOLOSCAN_LOG_INFO("Using Device memory location for output frames");
    }

    // Create pool of allocated frame buffers
    create_frame_pool();

    // Create converter and burst processor
    auto converter = PacketsToFramesConverter::create(this);
    burst_processor_ = std::make_unique<BurstProcessor>(std::move(converter));
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
  }

  /**
   * @brief Processes a received burst of packets and generates video frames.
   *
   * @param op_input The operator input context.
   * @param op_output The operator output context.
   * @param context The execution context.
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    BurstParams* burst;
    auto status = get_rx_burst(&burst, port_id_, parent_.queue_id_.get());
    if (status != Status::SUCCESS) return;

    const auto& packets_received = burst->hdr.hdr.num_pkts;
    total_packets_received_ += packets_received;
    if (packets_received == 0) { return; }
    if (total_packets_received_ > PACKETS_DISPLAY_INTERVAL) {
      HOLOSCAN_LOG_INFO("Got burst with {} pkts | total packets received {}",
                        packets_received,
                        total_packets_received_);
      total_packets_received_ = 0;
    }

    PACKET_TRACE_LOG("Processing burst: port={}, queue={}, packets={}, burst_ptr={}",
                     port_id_,
                     parent_.queue_id_.get(),
                     packets_received,
                     static_cast<void*>(burst));

    append_to_frame(burst);

    if (ready_frames_.empty()) return;

    size_t total_frames = ready_frames_.size();

    if (total_frames > 1) {
      HOLOSCAN_LOG_WARN(
          "Multiple frames ready ({}), dropping {} earlier frames to prevent pipeline issues",
          total_frames,
          total_frames - 1);
    }

    PACKET_TRACE_LOG("Ready frames count: {}, processing frame emission", total_frames);

    // Pop all frames but keep only the last one
    std::shared_ptr<FrameBufferBase> last_frame = nullptr;
    while (auto frame = pop_ready_frame()) {
      if (last_frame) {
        // Return the previous frame back to pool (dropping it)
        frames_pool_.push_back(last_frame);
        PACKET_TRACE_LOG("Dropped frame returned to pool: size={}, ptr={}",
                         last_frame->get_size(),
                         static_cast<void*>(last_frame->get()));
      }
      last_frame = frame;
    }

    // Emit only the last (most recent) frame
    if (last_frame) {
      PACKET_TRACE_LOG("Emitting latest frame: {} bytes", last_frame->get_size());
      PACKET_TRACE_LOG("Frame emission details: size={}, ptr={}, memory_location={}",
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
    size_t ready_frames_before = ready_frames_.size();

    PACKET_TRACE_LOG("Processing burst: ready_frames_before={}, queue_size={}, burst_packets={}",
                     ready_frames_before,
                     bursts_awaiting_cleanup_.size(),
                     burst->hdr.hdr.num_pkts);

    burst_processor_->process_burst(burst, parent_.hds_.get());

    size_t ready_frames_after = ready_frames_.size();

    PACKET_TRACE_LOG("Burst processed: ready_frames_after={}, frames_completed={}",
                     ready_frames_after,
                     ready_frames_after - ready_frames_before);

    // If new frames were completed, free all accumulated bursts
    if (ready_frames_after > ready_frames_before) {
      size_t frames_completed = ready_frames_after - ready_frames_before;
      PACKET_TRACE_LOG("{} frame(s) completed, freeing {} accumulated bursts",
                       frames_completed,
                       bursts_awaiting_cleanup_.size());
      PACKET_TRACE_LOG("Freeing accumulated bursts: count={}", bursts_awaiting_cleanup_.size());
      while (!bursts_awaiting_cleanup_.empty()) {
        auto burst_to_free = bursts_awaiting_cleanup_.front();
        PACKET_TRACE_LOG("Freeing burst: ptr={}, packets={}",
                         static_cast<void*>(burst_to_free),
                         burst_to_free->hdr.hdr.num_pkts);
        free_all_packets_and_burst_rx(burst_to_free);
        bursts_awaiting_cleanup_.pop_front();
      }
    }

    // Add current burst to the queue after freeing previous bursts
    bursts_awaiting_cleanup_.push_back(burst);

    PACKET_TRACE_LOG("Final queue_size={}, burst_ptr={}",
                     bursts_awaiting_cleanup_.size(),
                     static_cast<void*>(burst));
  }

  /**
   * @brief Retrieves a ready frame from the queue if available.
   *
   * @return Shared pointer to a ready frame, or nullptr if none is available.
   */
  std::shared_ptr<FrameBufferBase> pop_ready_frame() {
    if (ready_frames_.empty()) { return nullptr; }

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

  std::shared_ptr<FrameBufferBase> get_allocated_frame() override {
    if (frames_pool_.empty()) {
      throw std::runtime_error("Running out of resources, frames pool is empty");
    }
    auto frame = frames_pool_.front();
    frames_pool_.pop_front();
    return frame;
  }

  void on_new_frame(std::shared_ptr<FrameBufferBase> frame) override {
    ready_frames_.push_back(frame);
    PACKET_TRACE_LOG("New frame ready: {}", frame->get_size());
  }

 private:
  AdvNetworkMediaRxOp& parent_;
  int port_id_;
  std::unique_ptr<BurstProcessor> burst_processor_;
  std::deque<std::shared_ptr<FrameBufferBase>> frames_pool_;
  std::deque<std::shared_ptr<FrameBufferBase>> ready_frames_;
  std::deque<BurstParams*> bursts_awaiting_cleanup_;
  size_t total_packets_received_ = 0;
  nvidia::gxf::VideoFormat video_format_;
  size_t frame_size_;
  OutputFormatType output_format_{OutputFormatType::VIDEO_BUFFER};
  nvidia::gxf::MemoryStorageType storage_type_{nvidia::gxf::MemoryStorageType::kDevice};
};

AdvNetworkMediaRxOp::AdvNetworkMediaRxOp() : pimpl_(nullptr) {}

AdvNetworkMediaRxOp::~AdvNetworkMediaRxOp() {
  if (pimpl_) {
    pimpl_->cleanup();  // Clean up allocated resources
    delete pimpl_;
    pimpl_ = nullptr;
  }
}

void AdvNetworkMediaRxOp::setup(OperatorSpec& spec) {
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
}

void AdvNetworkMediaRxOp::initialize() {
  HOLOSCAN_LOG_INFO("AdvNetworkMediaRxOp::initialize()");
  holoscan::Operator::initialize();

  if (!pimpl_) { pimpl_ = new AdvNetworkMediaRxOpImpl(*this); }

  pimpl_->initialize();
}

void AdvNetworkMediaRxOp::compute(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext& context) {
  pimpl_->compute(op_input, op_output, context);
}

};  // namespace holoscan::ops
