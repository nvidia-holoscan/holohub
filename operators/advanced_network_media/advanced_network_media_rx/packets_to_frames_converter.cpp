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

#include <cuda_runtime.h>
#include "advanced_network/common.h"
#include "packets_to_frames_converter.h"
#include "../common/adv_network_media_common.h"

namespace holoscan::ops {

cudaMemcpyKind MemoryCopyHelper::get_copy_kind(nvidia::gxf::MemoryStorageType src_storage_type,
                                               nvidia::gxf::MemoryStorageType dst_storage_type) {
  // Determine copy kind based on source and destination combinations
  if (src_storage_type == nvidia::gxf::MemoryStorageType::kHost) {
    if (dst_storage_type == nvidia::gxf::MemoryStorageType::kHost) {
      return cudaMemcpyHostToHost;
    } else {
      return cudaMemcpyHostToDevice;
    }
  } else {  // nvidia::gxf::MemoryStorageType::kDevice
    if (dst_storage_type == nvidia::gxf::MemoryStorageType::kHost) {
      return cudaMemcpyDeviceToHost;
    } else {
      return cudaMemcpyDeviceToDevice;
    }
  }
}

nvidia::gxf::MemoryStorageType MemoryCopyHelper::to_storage_type(MemoryLocation location) {
  switch (location) {
    case MemoryLocation::Host:
      return nvidia::gxf::MemoryStorageType::kHost;
    case MemoryLocation::GPU:
      return nvidia::gxf::MemoryStorageType::kDevice;
    default:
      return nvidia::gxf::MemoryStorageType::kDevice;
  }
}

void ContiguousStrategy::process_packet(PacketsToFramesConverter& converter, uint8_t* payload,
                                        size_t payload_size) {
  // Validate input parameters to prevent crashes
  if (payload == nullptr) {
    HOLOSCAN_LOG_ERROR("ContiguousStrategy: Received null payload pointer! Skipping packet.");
    return;
  }
  if (payload_size == 0) {
    HOLOSCAN_LOG_ERROR("ContiguousStrategy: Received zero payload size! Skipping packet.");
    return;
  }

  if (current_payload_start_ptr_ == nullptr) {
    current_payload_start_ptr_ = payload;
    PACKET_TRACE_LOG("ContiguousStrategy: Starting new accumulation at {} (size={})",
                     static_cast<void*>(payload),
                     payload_size);
  }

  bool is_contiguous_memory = current_payload_start_ptr_ + accumulated_contiguous_size_ == payload;
  if (!is_contiguous_memory) {
    PACKET_TRACE_LOG("ContiguousStrategy: Stride break detected, copying {} bytes",
                     accumulated_contiguous_size_);
    // Execute copy for previous accumulated batch (this will advance frame position)
    execute_copy(converter);

    // Start new accumulation batch with current packet
    current_payload_start_ptr_ = payload;
    accumulated_contiguous_size_ = 0;
  }

  // CRITICAL FIX: Check if adding this packet would exceed frame boundaries
  // This prevents accumulation across multiple frames
  auto frame = converter.get_destination_frame_buffer();
  size_t frame_remaining = frame->get_size() - converter.get_frame_position();
  size_t new_total_size = accumulated_contiguous_size_ + payload_size;

  if (new_total_size > frame_remaining && accumulated_contiguous_size_ > 0) {
    PACKET_TRACE_LOG(
        "ContiguousStrategy: Frame boundary detected, executing copy of {} bytes before adding new "
        "packet",
        accumulated_contiguous_size_);
    // Execute copy for current accumulated data to avoid frame overflow
    execute_copy(converter);

    // Start new accumulation with current packet
    current_payload_start_ptr_ = payload;
    accumulated_contiguous_size_ = 0;
  }

  accumulated_contiguous_size_ += payload_size;
}

void ContiguousStrategy::execute_copy(PacketsToFramesConverter& converter) {
  if (accumulated_contiguous_size_ == 0 || current_payload_start_ptr_ == nullptr) return;

  auto frame = converter.get_destination_frame_buffer();
  uint8_t* dst_ptr = static_cast<uint8_t*>(frame->get()) + converter.get_frame_position();

  PACKET_TRACE_LOG(
      "ContiguousStrategy: Executing copy - frame_pos_before={}, copy_size={}, frame_total_size={}",
      converter.get_frame_position(),
      accumulated_contiguous_size_,
      frame->get_size());

  if (dst_ptr == nullptr) {
    HOLOSCAN_LOG_ERROR("ERROR: dst_ptr is NULL! Skipping copy operation.");
    return;
  }
  if (current_payload_start_ptr_ == nullptr) {
    HOLOSCAN_LOG_ERROR("ERROR: current_payload_start_ptr_ is NULL! Skipping copy operation.");
    return;
  }
  if (accumulated_contiguous_size_ == 0) {
    HOLOSCAN_LOG_ERROR("WARNING: accumulated_contiguous_size_ is 0! Skipping copy operation.");
    return;
  }

  size_t dst_offset = converter.get_frame_position();
  size_t frame_size = frame->get_size();
  if (dst_offset + accumulated_contiguous_size_ > frame_size) {
    HOLOSCAN_LOG_ERROR(
        "ERROR: Copy would exceed frame bounds! dst_offset={}, copy_size={}, frame_size={}. "
        "Skipping copy operation.",
        dst_offset,
        accumulated_contiguous_size_,
        frame_size);
    return;
  }

  CUDA_TRY(
      cudaMemcpy(dst_ptr, current_payload_start_ptr_, accumulated_contiguous_size_, copy_kind_));

  // Advance frame position by the amount we just copied
  converter.advance_frame_position(accumulated_contiguous_size_);

  PACKET_TRACE_LOG("ContiguousStrategy: Copy completed - frame_pos_after={}, copy_size={}",
                   converter.get_frame_position(),
                   accumulated_contiguous_size_);

  // Reset accumulation state for next batch
  current_payload_start_ptr_ = nullptr;
  accumulated_contiguous_size_ = 0;
}

void ContiguousStrategy::reset_state() {
  current_payload_start_ptr_ = nullptr;
  accumulated_contiguous_size_ = 0;
}

void StridedStrategy::process_packet(PacketsToFramesConverter& converter, uint8_t* payload,
                                     size_t payload_size) {
  if (payload == nullptr) {
    HOLOSCAN_LOG_ERROR("StridedStrategy: Received null payload pointer! Skipping packet.");
    return;
  }
  if (payload_size == 0) {
    HOLOSCAN_LOG_ERROR("StridedStrategy: Received zero payload size! Skipping packet.");
    return;
  }

  if (first_packet_ptr_ == nullptr) {
    reset_accumulation_state(payload, payload_size);

    HOLOSCAN_LOG_INFO("Strided strategy: First packet at {}, size: {}",
                      static_cast<void*>(payload),
                      payload_size);
    return;
  }

  if (!is_stride_maintained(payload, payload_size)) {
    HOLOSCAN_LOG_INFO("Stride pattern broken, executing accumulated copy and falling back");

    // Execute accumulated strided copy if we have multiple packets
    if (packet_count_ > 1) {
      execute_accumulated_strided_copy(converter);
    } else {
      // Single packet - use individual copy
      execute_individual_copy(converter, first_packet_ptr_, total_data_size_);
    }

    // Reset accumulation and start fresh with current packet
    reset_accumulation_state(payload, payload_size);
    return;
  }

  // Stride is maintained - continue accumulating
  last_packet_ptr_ = payload;
  packet_count_++;
  total_data_size_ += payload_size;
}

void StridedStrategy::execute_copy(PacketsToFramesConverter& converter) {
  if (packet_count_ == 0 || first_packet_ptr_ == nullptr) return;

  if (packet_count_ > 1 && stride_validated_) {
    execute_accumulated_strided_copy(converter);
  } else {
    execute_individual_copy(converter, first_packet_ptr_, total_data_size_);
  }
}

void StridedStrategy::execute_accumulated_strided_copy(PacketsToFramesConverter& converter) {
  if (packet_count_ <= 1 || first_packet_ptr_ == nullptr) return;

  auto frame = converter.get_destination_frame_buffer();
  uint8_t* dst_ptr = static_cast<uint8_t*>(frame->get()) + converter.get_frame_position();

  // Use cudaMemcpy2D for strided copy
  size_t width = stride_info_.payload_size;  // Width of each row (payload size)
  size_t height = packet_count_;             // Number of rows (packets)
  size_t src_pitch = actual_stride_;         // Actual detected stride
  size_t dst_pitch = width;                  // Destination pitch (contiguous)

  PACKET_TRACE_LOG("Executing strided copy: width={}, height={}, src_pitch={}, dst_pitch={}",
                   width,
                   height,
                   src_pitch,
                   dst_pitch);

  if (dst_ptr == nullptr) {
    HOLOSCAN_LOG_ERROR("ERROR: dst_ptr is NULL! Skipping strided copy operation.");
    return;
  }
  if (first_packet_ptr_ == nullptr) {
    HOLOSCAN_LOG_ERROR("ERROR: first_packet_ptr_ is NULL! Skipping strided copy operation.");
    return;
  }

  // Check bounds to prevent frame overflow
  size_t total_copy_size = width * height;
  size_t dst_offset = converter.get_frame_position();
  size_t frame_size = frame->get_size();
  if (dst_offset + total_copy_size > frame_size) {
    HOLOSCAN_LOG_ERROR(
        "ERROR: Strided copy would exceed frame bounds! dst_offset={}, copy_size={}, "
        "frame_size={}. Skipping copy operation.",
        dst_offset,
        total_copy_size,
        frame_size);
    return;
  }

  CUDA_TRY(
      cudaMemcpy2D(dst_ptr, dst_pitch, first_packet_ptr_, src_pitch, width, height, copy_kind_));

  converter.advance_frame_position(total_data_size_);
}

void StridedStrategy::execute_individual_copy(PacketsToFramesConverter& converter,
                                              uint8_t* payload_ptr, size_t payload_size) {
  auto frame = converter.get_destination_frame_buffer();
  uint8_t* dst_ptr = static_cast<uint8_t*>(frame->get()) + converter.get_frame_position();

  PACKET_TRACE_LOG("Executing individual copy: size={}", payload_size);

  if (dst_ptr == nullptr) {
    HOLOSCAN_LOG_ERROR("ERROR: dst_ptr is NULL! Skipping individual copy operation.");
    return;
  }
  if (payload_ptr == nullptr) {
    HOLOSCAN_LOG_ERROR("ERROR: payload_ptr is NULL! Skipping individual copy operation.");
    return;
  }
  if (payload_size == 0) {
    HOLOSCAN_LOG_ERROR("WARNING: payload_size is 0! Skipping individual copy operation.");
    return;
  }

  // Check bounds to prevent frame overflow
  size_t dst_offset = converter.get_frame_position();
  size_t frame_size = frame->get_size();
  if (dst_offset + payload_size > frame_size) {
    HOLOSCAN_LOG_ERROR(
        "ERROR: Individual copy would exceed frame bounds! dst_offset={}, copy_size={}, "
        "frame_size={}. Skipping copy operation.",
        dst_offset,
        payload_size,
        frame_size);
    return;
  }

  CUDA_TRY(cudaMemcpy(dst_ptr, payload_ptr, payload_size, copy_kind_));

  converter.advance_frame_position(payload_size);
}

bool StridedStrategy::is_stride_maintained(uint8_t* payload, size_t payload_size) {
  if (!last_packet_ptr_) {
    // First packet in accumulation
    return true;
  }

  // Check if stride is maintained
  size_t actual_diff = payload - last_packet_ptr_;

  if (!stride_validated_) {
    // First stride validation
    actual_stride_ = actual_diff;
    stride_validated_ = true;

    // Verify it matches expected stride (with some tolerance)
    if (actual_stride_ != stride_info_.stride_size) {
      HOLOSCAN_LOG_INFO("Stride differs from expected: actual={}, expected={}",
                        actual_stride_,
                        stride_info_.stride_size);
      // Still continue with actual stride, but log the difference
    }

    HOLOSCAN_LOG_INFO("Stride validated: actual_stride={}", actual_stride_);
    return true;
  } else {
    // Subsequent stride validation
    if (actual_diff != actual_stride_) {
      HOLOSCAN_LOG_INFO(
          "Stride consistency broken: expected={}, actual={}", actual_stride_, actual_diff);
      return false;
    }
    return true;
  }
}

bool StridedStrategy::detect_buffer_wraparound(uint8_t* current_ptr, uint8_t* expected_ptr) const {
  // If current pointer is significantly lower than expected, likely a wrap-around
  if (current_ptr < expected_ptr) {
    ptrdiff_t backward_diff = expected_ptr - current_ptr;
    if (backward_diff > WRAPAROUND_THRESHOLD) { return true; }
  }

  // Additional check: if current pointer is way ahead of expected, might also indicate wrap
  if (current_ptr > expected_ptr) {
    ptrdiff_t forward_diff = current_ptr - expected_ptr;
    if (forward_diff > WRAPAROUND_THRESHOLD) { return true; }
  }

  return false;
}

void StridedStrategy::reset_state() {
  first_packet_ptr_ = nullptr;
  last_packet_ptr_ = nullptr;
  packet_count_ = 0;
  total_data_size_ = 0;
  stride_validated_ = false;
  actual_stride_ = 0;
}

void StridedStrategy::reset_accumulation_state(uint8_t* payload, size_t payload_size) {
  first_packet_ptr_ = payload;
  last_packet_ptr_ = payload;
  packet_count_ = 1;
  total_data_size_ = payload_size;
  stride_validated_ = false;
  actual_stride_ = 0;
}

void PacketCopyStrategyDetector::configure_burst_parameters(size_t header_stride_size,
                                                            size_t payload_stride_size,
                                                            bool hds_on) {
  if (!strategy_confirmed_) {
    if (packets_analyzed_ > 0) {
      bool config_changed =
          (expected_header_stride_ != header_stride_size ||
           expected_payload_stride_ != payload_stride_size || expected_hds_on_ != hds_on);

      if (config_changed) {
        HOLOSCAN_LOG_WARN("Burst configuration changed during detection, restarting analysis");
        reset_detection();
      }
    }

    expected_header_stride_ = header_stride_size;
    expected_payload_stride_ = payload_stride_size;
    expected_hds_on_ = hds_on;

    HOLOSCAN_LOG_INFO(
        "Burst info set for detection - header_stride: {}, payload_stride: {}, hds_on: {}",
        header_stride_size,
        payload_stride_size,
        hds_on);
  } else {
    HOLOSCAN_LOG_INFO("Strategy already confirmed, ignoring burst info update");
  }
}

bool PacketCopyStrategyDetector::collect_packet_info(const RtpParams& rtp_params, uint8_t* payload,
                                                     size_t payload_size) {
  if (strategy_confirmed_) { return true; }

  // Use provided RTP parameters directly (no parsing needed)
  detection_payloads_.push_back(payload);
  detection_payload_sizes_.push_back(payload_size);
  detection_rtp_sequences_.push_back(rtp_params.sequence_number);
  packets_analyzed_++;

  HOLOSCAN_LOG_INFO("Collected packet {} for detection: payload={}, size={}, seq={}",
                    packets_analyzed_,
                    static_cast<void*>(payload),
                    payload_size,
                    rtp_params.sequence_number);

  return packets_analyzed_ >= STRATEGY_DETECTION_PACKETS;
}

std::unique_ptr<IPacketCopyStrategy> PacketCopyStrategyDetector::detect_strategy() {
  if (detection_payloads_.size() < 2) {
    strategy_confirmed_ = true;
    HOLOSCAN_LOG_INFO("Insufficient packets for analysis, defaulting to CONTIGUOUS strategy");
    return std::make_unique<ContiguousStrategy>();
  }

  if (!validate_rtp_sequence_continuity()) {
    HOLOSCAN_LOG_WARN("RTP sequence drops detected, restarting detection");
    reset_detection();
    return nullptr;
  }

  if (detect_cyclic_buffer_wraparound()) {
    HOLOSCAN_LOG_WARN("Cyclic buffer wrap-around detected, restarting detection");
    reset_detection();
    return nullptr;
  }

  auto analysis_result = analyze_packet_pattern();
  if (!analysis_result) {
    HOLOSCAN_LOG_WARN("Pattern analysis failed, restarting detection");
    reset_detection();
    return nullptr;
  }

  strategy_confirmed_ = true;

  auto [strategy_type, stride_info] = *analysis_result;

  HOLOSCAN_LOG_INFO("Strategy detection completed: {} (stride: {}, payload: {})",
                    strategy_type == CopyStrategy::CONTIGUOUS ? "CONTIGUOUS" : "STRIDED",
                    stride_info.stride_size,
                    stride_info.payload_size);

  if (strategy_type == CopyStrategy::CONTIGUOUS) {
    return std::make_unique<ContiguousStrategy>();
  } else {
    return std::make_unique<StridedStrategy>(stride_info);
  }
}

void PacketCopyStrategyDetector::reset_detection() {
  strategy_confirmed_ = false;
  detection_payloads_.clear();
  detection_payload_sizes_.clear();
  detection_rtp_sequences_.clear();
  packets_analyzed_ = 0;

  HOLOSCAN_LOG_INFO("Strategy detection reset");
}

std::optional<std::pair<CopyStrategy, StrideInfo>>
PacketCopyStrategyDetector::analyze_packet_pattern() {
  if (detection_payloads_.size() < 2) return std::nullopt;

  size_t expected_stride = expected_hds_on_ ? expected_payload_stride_
                                            : (expected_header_stride_ + expected_payload_stride_);

  // Analyze pointer differences for exact contiguity and stride consistency
  bool is_exactly_contiguous = true;
  bool is_stride_consistent = true;
  size_t actual_stride = 0;
  size_t detected_payload_size = detection_payload_sizes_[0];  // Use first packet's payload size

  // Verify all packets have exactly the same payload size
  for (size_t i = 1; i < detection_payload_sizes_.size(); ++i) {
    if (detection_payload_sizes_[i] != detected_payload_size) {
      HOLOSCAN_LOG_WARN("Inconsistent payload sizes detected: first={}, packet_{}={}",
                        detected_payload_size,
                        i,
                        detection_payload_sizes_[i]);
      return std::nullopt;  // Cannot handle varying payload sizes
    }
  }

  for (size_t i = 1; i < detection_payloads_.size(); ++i) {
    uint8_t* prev_ptr = detection_payloads_[i - 1];
    uint8_t* curr_ptr = detection_payloads_[i];

    size_t pointer_diff = curr_ptr - prev_ptr;

    if (i == 1) { actual_stride = pointer_diff; }

    // Check for exact contiguity: next pointer should be exactly at previous_pointer +
    // previous_payload_size
    uint8_t* expected_next_ptr = prev_ptr + detected_payload_size;

    if (curr_ptr != expected_next_ptr) {
      is_exactly_contiguous = false;
      HOLOSCAN_LOG_INFO("Non-contiguous detected: packet {}, expected ptr {}, actual ptr {}",
                        i,
                        static_cast<void*>(expected_next_ptr),
                        static_cast<void*>(curr_ptr));
    }

    // Check if stride is consistent (independent of contiguity)
    if (pointer_diff != actual_stride) {
      is_stride_consistent = false;
      HOLOSCAN_LOG_INFO("Inconsistent stride detected: packet {}, expected stride {}, actual {}",
                        i,
                        actual_stride,
                        pointer_diff);
      break;  // No point checking further if stride is inconsistent
    }
  }

  StrideInfo stride_info;
  stride_info.stride_size = actual_stride;
  stride_info.payload_size = detected_payload_size;  // Use exact payload size, not average

  // Three-way decision logic:
  CopyStrategy strategy;
  if (is_exactly_contiguous) {
    // Case 1: Exactly contiguous - use efficient CONTIGUOUS strategy
    strategy = CopyStrategy::CONTIGUOUS;
    HOLOSCAN_LOG_INFO("Packets are exactly contiguous, using CONTIGUOUS strategy");
  } else if (is_stride_consistent) {
    // Case 2: Gaps but consistent stride - use STRIDED strategy
    strategy = CopyStrategy::STRIDED;
    HOLOSCAN_LOG_INFO(
        "Packets have gaps but consistent stride, using STRIDED strategy (stride: {}, payload: {})",
        actual_stride,
        detected_payload_size);
  } else {
    // Case 3: Gaps with inconsistent stride - fallback to CONTIGUOUS (individual copies)
    strategy = CopyStrategy::CONTIGUOUS;
    HOLOSCAN_LOG_INFO(
        "Packets have inconsistent stride, falling back to CONTIGUOUS strategy for individual "
        "copies");
  }

  HOLOSCAN_LOG_INFO(
      "Analysis results - actual_stride: {}, expected_stride: {}, payload_size: {}, "
      "exactly_contiguous: {}, stride_consistent: {}",
      actual_stride,
      expected_stride,
      detected_payload_size,
      is_exactly_contiguous,
      is_stride_consistent);

  return std::make_pair(strategy, stride_info);
}

bool PacketCopyStrategyDetector::validate_rtp_sequence_continuity() const {
  if (detection_rtp_sequences_.size() < 2) { return true; }

  for (size_t i = 1; i < detection_rtp_sequences_.size(); ++i) {
    uint64_t prev_seq = detection_rtp_sequences_[i - 1];
    uint64_t curr_seq = detection_rtp_sequences_[i];

    // Handle 16-bit sequence number wrap-around
    uint64_t expected_seq = (prev_seq + 1) & 0xFFFFFFFFFFFFFFFF;

    if (curr_seq != expected_seq) {
      HOLOSCAN_LOG_INFO("RTP sequence discontinuity: expected {}, got {} (prev was {})",
                        expected_seq,
                        curr_seq,
                        prev_seq);
      return false;
    }
  }

  return true;
}

bool PacketCopyStrategyDetector::detect_cyclic_buffer_wraparound() const {
  if (detection_payloads_.size() < 2) { return false; }

  for (size_t i = 1; i < detection_payloads_.size(); ++i) {
    uint8_t* prev_ptr = detection_payloads_[i - 1];
    uint8_t* curr_ptr = detection_payloads_[i];

    // If current pointer is significantly lower than previous, likely a wrap-around
    if (curr_ptr < prev_ptr) {
      ptrdiff_t backward_diff = prev_ptr - curr_ptr;
      // Consider it a wrap-around if the backward difference is large (> 1MB)
      if (backward_diff > 1024 * 1024) {
        HOLOSCAN_LOG_INFO("Potential cyclic buffer wrap-around detected: {} -> {}",
                          static_cast<void*>(prev_ptr),
                          static_cast<void*>(curr_ptr));
        return true;
      }
    }
  }

  return false;
}

PacketsToFramesConverter::PacketsToFramesConverter(std::shared_ptr<IFrameProvider> frame_provider)
    : frame_provider_(frame_provider),
      frame_(frame_provider->get_allocated_frame()),
      waiting_for_end_of_frame_(false),
      current_byte_in_frame_(0) {
  detector_.reset_detection();
}

std::unique_ptr<PacketsToFramesConverter> PacketsToFramesConverter::create(
    IFrameProvider* provider) {
  // Create a non-owning shared_ptr wrapper
  // This is safe as long as the provider outlives the converter (which it does in our ownership
  // model)
  auto shared_provider = std::shared_ptr<IFrameProvider>(provider, [](IFrameProvider*) {
    // Custom deleter that does nothing - we don't own the pointer
  });

  return std::make_unique<PacketsToFramesConverter>(shared_provider);
}

void PacketsToFramesConverter::configure_burst_parameters(size_t header_stride_size,
                                                          size_t payload_stride_size, bool hds_on) {
  detector_.configure_burst_parameters(header_stride_size, payload_stride_size, hds_on);
}

void PacketsToFramesConverter::set_source_memory_location(bool payload_on_cpu) {
  source_memory_location_ = payload_on_cpu ? nvidia::gxf::MemoryStorageType::kHost
                                           : nvidia::gxf::MemoryStorageType::kDevice;

  // Update current strategy if it exists
  if (current_strategy_) {
    auto dst_location = MemoryCopyHelper::to_storage_type(frame_->get_memory_location());
    current_strategy_->set_memory_locations(source_memory_location_, dst_location);
  }

  HOLOSCAN_LOG_INFO(
      "Source memory location set to: {} (payload_on_cpu={})",
      source_memory_location_ == nvidia::gxf::MemoryStorageType::kHost ? "HOST" : "DEVICE",
      payload_on_cpu);
}

void PacketsToFramesConverter::set_source_memory_location(
    nvidia::gxf::MemoryStorageType src_storage_type) {
  source_memory_location_ = src_storage_type;

  // Update current strategy if it exists
  if (current_strategy_) {
    auto dst_location = MemoryCopyHelper::to_storage_type(frame_->get_memory_location());
    current_strategy_->set_memory_locations(source_memory_location_, dst_location);
  }

  HOLOSCAN_LOG_INFO(
      "Source memory location set to: {}",
      source_memory_location_ == nvidia::gxf::MemoryStorageType::kHost ? "HOST" : "DEVICE");
}

void PacketsToFramesConverter::process_incoming_packet(const RtpParams& rtp_params,
                                                       uint8_t* payload) {
  if (waiting_for_end_of_frame_) {
    if (rtp_params.m_bit) {
      HOLOSCAN_LOG_INFO("End of frame received, restarting");
      waiting_for_end_of_frame_ = false;
      reset_frame_state();
    }
    return;
  }

  // Strategy detection phase (only during initialization)
  if (!current_strategy_) {
    if (!handle_strategy_detection(rtp_params, payload, rtp_params.payload_size)) {
      return;  // Still detecting, skip processing
    }
    ensure_strategy_available();
  }

  // Check if this is the end-of-frame marker packet BEFORE processing
  if (rtp_params.m_bit) {
    PACKET_TRACE_LOG(
        "End-of-frame marker detected (M-bit=1) for seq={}, payload_size={}, frame_pos={}",
        rtp_params.sequence_number,
        rtp_params.payload_size,
        current_byte_in_frame_);

    // CRITICAL: Execute any pending copy operations from previous packets FIRST
    // This ensures we don't mix data from multiple frames in the same copy operation
    if (current_strategy_ && current_strategy_->has_pending_copy()) {
      PACKET_TRACE_LOG(
          "Executing pending copy operations before processing M-bit packet (frame_pos={})",
          current_byte_in_frame_);
      current_strategy_->execute_copy(*this);
      PACKET_TRACE_LOG("Pending copy completed, frame_pos now={}", current_byte_in_frame_);
    }

    // Now process the final packet data for this frame
    current_strategy_->process_packet(*this, payload, rtp_params.payload_size);
    // Then handle end of frame
    handle_end_of_frame();
    return;
  }

  // Main data path - process packet (only for non-marker packets)
  current_strategy_->process_packet(*this, payload, rtp_params.payload_size);

  // Validate packet integrity (only for non-marker packets)
  const auto& [is_corrupted, error] = validate_packet_integrity(rtp_params);
  if (is_corrupted) {
    HOLOSCAN_LOG_ERROR("Frame is corrupted: {}", error);
    waiting_for_end_of_frame_ = true;  // Wait for next frame start
  }
}

void PacketsToFramesConverter::reset_frame_state() {
  current_byte_in_frame_ = 0;
  if (current_strategy_) { current_strategy_->reset_state(); }
}

CopyStrategy PacketsToFramesConverter::get_current_strategy() const {
  if (current_strategy_) { return current_strategy_->get_strategy_type(); }
  return CopyStrategy::UNKNOWN;
}

void PacketsToFramesConverter::force_strategy_redetection() {
  detector_.reset_detection();
  current_strategy_.reset();
  HOLOSCAN_LOG_INFO("Forced strategy re-detection");
}

bool PacketsToFramesConverter::has_pending_copy() const {
  if (!current_strategy_) {
    HOLOSCAN_LOG_DEBUG("has_pending_copy: no strategy, returning false");
    return false;
  }
  bool result = current_strategy_->has_pending_copy();
  return result;
}

std::pair<bool, std::string> PacketsToFramesConverter::validate_packet_integrity(
    const RtpParams& rtp_params) {
  int64_t bytes_left = frame_->get_size() - current_byte_in_frame_;

  if (bytes_left < 0) {
    return {true, "Frame received is not aligned to the frame size and will be dropped"};
  }

  bool frame_full = bytes_left <= 0;
  if (frame_full && !rtp_params.m_bit) {
    return {true, "Frame is full but marker was not not appear"};
  }

  if (!frame_full && rtp_params.m_bit) {
    HOLOSCAN_LOG_ERROR(
        "Marker appeared but frame is not full - Frame details: "
        "current_position={} bytes, total_frame_size={} bytes, "
        "bytes_remaining={} bytes, payload_size={} bytes, "
        "sequence_number={}",
        current_byte_in_frame_,
        frame_->get_size(),
        bytes_left,
        rtp_params.payload_size,
        rtp_params.sequence_number);
    return {true, "Marker appeared but frame is not full"};
  }

  return {false, ""};
}

bool PacketsToFramesConverter::handle_strategy_detection(const RtpParams& rtp_params,
                                                         uint8_t* payload, size_t payload_size) {
  if (detector_.is_strategy_confirmed()) {
    return true;  // Strategy already confirmed
  }

  HOLOSCAN_LOG_TRACE("Processing packet for detection: seq={}, payload_size={}, m_bit={}",
                     rtp_params.sequence_number,
                     payload_size,
                     rtp_params.m_bit);

  bool ready_for_detection = detector_.collect_packet_info(rtp_params, payload, payload_size);
  if (ready_for_detection) {
    auto strategy = detector_.detect_strategy();
    if (strategy) {
      current_strategy_ = std::move(strategy);
      auto dst_location = MemoryCopyHelper::to_storage_type(frame_->get_memory_location());
      current_strategy_->set_memory_locations(source_memory_location_, dst_location);
      HOLOSCAN_LOG_INFO("Strategy detection successful: {}",
                        current_strategy_->get_strategy_type() == CopyStrategy::CONTIGUOUS
                            ? "CONTIGUOUS"
                            : "STRIDED");
      return true;  // Strategy ready
    } else {
      HOLOSCAN_LOG_INFO("Strategy detection failed, will retry with next packets");
    }
  } else {
    HOLOSCAN_LOG_INFO(
        "Still collecting packets for strategy detection, skipping packet processing");
  }

  return false;  // Still detecting
}

void PacketsToFramesConverter::ensure_strategy_available() {
  if (!current_strategy_) {
    current_strategy_ = std::make_unique<ContiguousStrategy>();
    auto dst_location = MemoryCopyHelper::to_storage_type(frame_->get_memory_location());
    current_strategy_->set_memory_locations(source_memory_location_, dst_location);
    HOLOSCAN_LOG_INFO("Using fallback CONTIGUOUS strategy");
  }
}

void PacketsToFramesConverter::handle_end_of_frame() {
  PACKET_TRACE_LOG(
      "End of frame marker received, executing final copy. Frame position: {}/{} bytes",
      current_byte_in_frame_,
      frame_->get_size());

  // Execute any remaining accumulated data
  current_strategy_->execute_copy(*this);

  // Validate frame completion
  int64_t bytes_left = frame_->get_size() - current_byte_in_frame_;
  if (bytes_left > 0) {
    HOLOSCAN_LOG_WARN("Frame incomplete after marker: {} bytes missing (expected: {}, actual: {})",
                      bytes_left,
                      frame_->get_size(),
                      current_byte_in_frame_);
  } else if (bytes_left < 0) {
    HOLOSCAN_LOG_ERROR("Frame overflow after marker: {} bytes too many (expected: {}, actual: {})",
                       -bytes_left,
                       frame_->get_size(),
                       current_byte_in_frame_);
  } else {
    PACKET_TRACE_LOG("Frame completed successfully: {} bytes", current_byte_in_frame_);
  }

  // Complete frame and get new one
  frame_provider_->on_new_frame(frame_);
  frame_ = frame_provider_->get_allocated_frame();
  reset_frame_state();
}

}  // namespace holoscan::ops
