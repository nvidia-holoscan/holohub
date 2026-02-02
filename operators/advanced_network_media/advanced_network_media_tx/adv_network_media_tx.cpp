/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdexcept>
#include "../common/frame_buffer.h"
#include "../common/video_parameters.h"
#include "../common/adv_network_media_logging.h"
#include "adv_network_media_tx.h"

namespace holoscan::ops {

namespace ano = holoscan::advanced_network;

using holoscan::advanced_network::BurstParams;
using holoscan::advanced_network::Status;

/**
 * @class AdvNetworkMediaTxOpImpl
 * @brief Implementation class for the AdvNetworkMediaTxOp operator.
 *
 * Handles the actual processing of media frames and communication with
 * the network infrastructure.
 */
class AdvNetworkMediaTxOpImpl {
 public:
  static constexpr int DISPLAY_WARNING_AFTER_BURST_NOT_AVAILABLE = 1000;
  static constexpr int MAX_RETRY_ATTEMPTS_BEFORE_DROP = 10;
  static constexpr int SLEEP_WHEN_BURST_NOT_AVAILABLE_US = 100;

  /**
   * @brief Constructs an implementation for the given operator.
   *
   * @param parent Reference to the parent operator.
   */
  explicit AdvNetworkMediaTxOpImpl(AdvNetworkMediaTxOp& parent) : parent_(parent) {}

  /**
   * @brief Initializes the implementation.
   *
   * Sets up the network port, calculates frame size, and prepares
   * for media transmission.
   */
  void initialize() {
    ANM_LOG_INFO("AdvNetworkMediaTxOp::initialize()");
    try {
      port_id_ = ano::get_port_id(parent_.interface_name_.get());
      if (port_id_ == -1) {
        ANM_LOG_ERROR("Invalid TX port {} specified in the config",
          parent_.interface_name_.get());
        throw std::runtime_error(
          "Invalid TX port '" + std::string(parent_.interface_name_.get()) +
          "' specified in the config (port_id=" + std::to_string(port_id_) + ")");
      } else {
        ANM_CONFIG_LOG("TX port {} found", port_id_);
      }

      video_sampling_ = get_video_sampling_format(parent_.video_format_.get());
      color_bit_depth_ = get_color_bit_depth(parent_.bit_depth_.get());
      frame_size_ = calculate_frame_size(parent_.frame_width_.get(), parent_.frame_height_.get(),
        video_sampling_, color_bit_depth_);
      ANM_CONFIG_LOG("Expected frame size: {} bytes", frame_size_);

      expected_video_format_ = get_expected_gxf_video_format(video_sampling_, color_bit_depth_);

      ANM_LOG_INFO("AdvNetworkMediaTxOp::initialize() complete");
    } catch (const std::exception& e) {
      ANM_LOG_ERROR("Error in AdvNetworkMediaTxOp initialization: {}", e.what());
      throw;
    }
  }

  /**
   * @brief Creates a MediaFrame from a GXF entity containing a VideoBuffer.
   *
   * @param entity The GXF entity containing the video buffer.
   * @return A shared pointer to the created MediaFrame, or nullptr if validation fails.
   */
  std::shared_ptr<MediaFrame> create_media_frame_from_video_buffer(nvidia::gxf::Entity entity) {
    try {
      auto frame = std::make_unique<VideoBufferFrameBuffer>(std::move(entity));
      auto result = frame->validate_frame_parameters(parent_.frame_width_.get(),
        parent_.frame_height_.get(), frame_size_, expected_video_format_);

      if (result != Status::SUCCESS) {
        ANM_LOG_ERROR("Video buffer validation failed");
        return nullptr;
      }

      return std::make_shared<MediaFrame>(std::move(frame));
    } catch (const std::exception& e) {
      ANM_LOG_ERROR("Video buffer error: {}", e.what());
      return nullptr;
    }
  }

  /**
   * @brief Creates a MediaFrame from a GXF entity containing a Tensor.
   *
   * @param entity The GXF entity containing the tensor.
   * @return A shared pointer to the created MediaFrame, or nullptr if validation fails.
   */
  std::shared_ptr<MediaFrame> create_media_frame_from_tensor(nvidia::gxf::Entity entity) {
    try {
      auto frame = std::make_unique<TensorFrameBuffer>(std::move(entity), expected_video_format_);
      auto result = frame->validate_frame_parameters(parent_.frame_width_.get(),
        parent_.frame_height_.get(), frame_size_, expected_video_format_);

      if (result != Status::SUCCESS) {
        ANM_LOG_ERROR("Tensor validation failed");
        return nullptr;
      }

      return std::make_shared<MediaFrame>(std::move(frame));
    } catch (const std::exception& e) {
      ANM_LOG_ERROR("Tensor error: {}", e.what());
      return nullptr;
    }
  }

  /**
   * @brief Processes input data from the operator context.
   *
   * Extracts the GXF entity from the input and creates a MediaFrame for transmission.
   *
   * @param op_input The operator input context.
   */
  void process_input(InputContext& op_input) {
    // Check if we still have a pending frame from previous call
    if (pending_tx_frame_) {
      retry_attempts_++;

      // Check if we've exceeded maximum retry attempts - drop frame to prevent stall
      if (retry_attempts_ >= MAX_RETRY_ATTEMPTS_BEFORE_DROP) {
        dropped_frames_++;
        ANM_LOG_ERROR(
            "TX port {}, queue {} exceeded max retry attempts ({}). Dropping frame to prevent "
            "pipeline stall. Total dropped: {}",
            port_id_,
            parent_.queue_id_.get(),
            retry_attempts_,
            dropped_frames_);
        pending_tx_frame_ = nullptr;  // Drop the stuck frame
        retry_attempts_ = 0;
        // Fall through to receive new frame
      } else {
        // Frame still pending, skip new input this cycle and let process_output try again
        ANM_STATS_TRACE(
            "TX queue {} on port {} still has pending frame (retry_attempts: {}); "
            "skipping new input.",
            parent_.queue_id_.get(), port_id_, retry_attempts_);
        return;
      }
    }

    // No pending frame (or just dropped one), receive new input
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) return;

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    auto maybe_video_buffer = entity.get<nvidia::gxf::VideoBuffer>();
    if (maybe_video_buffer) {
      pending_tx_frame_ = create_media_frame_from_video_buffer(std::move(entity));
    } else {
      auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
      if (!maybe_tensor) {
        ANM_LOG_ERROR("Neither VideoBuffer nor Tensor found in message");
        return;
      }
      pending_tx_frame_ = create_media_frame_from_tensor(std::move(entity));
    }

    if (!pending_tx_frame_) {
      ANM_LOG_ERROR("Failed to create media frame");
      return;
    }

    // New frame received, reset retry counter
    retry_attempts_ = 0;
  }

  /**
   * @brief Processes output data for the operator context.
   *
   * Transmits the pending media frame over the network if available.
   *
   * @param op_output The operator output context.
   */
  void process_output(OutputContext& op_output) {
    if (!pending_tx_frame_) {
      ANM_LOG_ERROR("No pending TX frame");
      return;
    }

    if (!cur_msg_) {
      cur_msg_ = ano::create_tx_burst_params();
      ano::set_header(cur_msg_, port_id_, parent_.queue_id_.get(), 1, 1);
    }

    if (!ano::is_tx_burst_available(cur_msg_)) {
      std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_WHEN_BURST_NOT_AVAILABLE_US));
      if (++not_available_count_ == DISPLAY_WARNING_AFTER_BURST_NOT_AVAILABLE) {
        ANM_LOG_ERROR(
            "TX port {}, queue {}, burst not available too many times consecutively. "
            "Make sure memory region has enough buffers. Sent {} and error {}",
            port_id_,
            parent_.queue_id_.get(),
            sent_,
            err_);
        not_available_count_ = 0;
        err_++;
      }
      return;
    }
    not_available_count_ = 0;
    Status ret;
    if ((ret = ano::get_tx_packet_burst(cur_msg_)) != Status::SUCCESS) {
      ANM_LOG_ERROR("Error returned from get_tx_packet_burst: {}", static_cast<int>(ret));
      return;
    }

    cur_msg_->custom_pkt_data = std::move(pending_tx_frame_);
    pending_tx_frame_ = nullptr;

    ret = ano::send_tx_burst(cur_msg_);
    if (ret != Status::SUCCESS) {
      ANM_LOG_ERROR("Error returned from send_tx_burst: {}", static_cast<int>(ret));
      ano::free_tx_burst(cur_msg_);
      err_++;
    } else {
      sent_++;
    }
    cur_msg_ = nullptr;
    ANM_STATS_TRACE("AdvNetworkMediaTxOp::process_output() {}:{} done. Emitted{}/Error{}",
                       port_id_,
                       parent_.queue_id_.get(),
                       sent_,
                       err_);
  }

  BurstParams* cur_msg_ = nullptr;
  std::shared_ptr<MediaFrame> pending_tx_frame_ = nullptr;
  size_t frame_size_;
  nvidia::gxf::VideoFormat expected_video_format_;
  int port_id_;
  VideoFormatSampling video_sampling_;
  VideoColorBitDepth color_bit_depth_;

  // Per-instance state for frame handling
  int retry_attempts_ = 0;        // Number of consecutive attempts to send current frame
  int dropped_frames_ = 0;        // Total number of frames dropped due to max retry exceeded
  int not_available_count_ = 0;   // Count of consecutive burst not available
  int sent_ = 0;                  // Total number of frames successfully sent
  int err_ = 0;                   // Total number of errors encountered

 private:
  AdvNetworkMediaTxOp& parent_;
};

AdvNetworkMediaTxOp::AdvNetworkMediaTxOp() : pimpl_(nullptr) {
}

AdvNetworkMediaTxOp::~AdvNetworkMediaTxOp() {
  if (pimpl_) {
    delete pimpl_;
    pimpl_ = nullptr;
  }
}

void AdvNetworkMediaTxOp::initialize() {
  ANM_LOG_INFO("AdvNetworkMediaTxOp::initialize()");
  holoscan::Operator::initialize();

  if (!pimpl_) {
    pimpl_ = std::make_unique<AdvNetworkMediaTxOpImpl>(*this);
  }

  pimpl_->initialize();
}

void AdvNetworkMediaTxOp::setup(OperatorSpec& spec) {
  spec.input<nvidia::gxf::Entity>("input");
  spec.param<std::string>(interface_name_,
    "interface_name",
    "Name of NIC from advanced_network config",
    "Name of NIC from advanced_network config");
  spec.param<uint16_t>(queue_id_, "queue_id", "Queue ID", "Queue ID", default_queue_id);
  spec.param<uint32_t>(frame_width_, "frame_width", "Frame width", "Width of the frame", 1920);
  spec.param<uint32_t>(
    frame_height_, "frame_height", "Frame height", "Height of the frame", 1080);
  spec.param<uint32_t>(bit_depth_, "bit_depth", "Bit depth", "Number of bits per pixel", 8);
  spec.param(
      video_format_, "video_format", "Video Format", "Video sample format", std::string("RGB888"));
}

void AdvNetworkMediaTxOp::compute(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext& context) {
  pimpl_->process_input(op_input);
  pimpl_->process_output(op_output);
}

}  // namespace holoscan::ops
