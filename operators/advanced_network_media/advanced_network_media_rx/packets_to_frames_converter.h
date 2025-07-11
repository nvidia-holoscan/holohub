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

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_PACKETS_TO_FRAMES_CONVERTER_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_PACKETS_TO_FRAMES_CONVERTER_H_

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "advanced_network/common.h"
#include "../common/adv_network_media_common.h"
#include "../common/frame_buffer.h"

namespace holoscan::ops {

/**
 * @brief Strategy for copying packet data to frames
 */
enum class CopyStrategy {
  UNKNOWN,     // Strategy not yet determined
  CONTIGUOUS,  // Packets are exactly contiguous in memory - use cudaMemcpy
  STRIDED      // Packets have any gaps - use cudaMemcpy2D
};

/**
 * @brief Memory copy configuration helper
 */
class MemoryCopyHelper {
 public:
  /**
   * @brief Determine appropriate cudaMemcpyKind based on source and destination memory locations
   * @param src_storage_type Source memory storage type
   * @param dst_storage_type Destination memory storage type
   * @return Appropriate cudaMemcpyKind for the operation
   */
  static cudaMemcpyKind get_copy_kind(nvidia::gxf::MemoryStorageType src_storage_type,
                                      nvidia::gxf::MemoryStorageType dst_storage_type);

  /**
   * @brief Convert MemoryLocation to MemoryStorageType
   * @param location Memory location to convert
   * @return Corresponding MemoryStorageType
   */
  static nvidia::gxf::MemoryStorageType to_storage_type(MemoryLocation location);
};

/**
 * @brief Information about detected stride pattern
 */
struct StrideInfo {
  size_t stride_size = 0;   // Detected stride size between packets
  size_t payload_size = 0;  // Expected payload size per packet
};

/**
 * @brief Forward declarations
 */
class PacketsToFramesConverter;

/**
 * @brief Interface for providing and managing frame buffers
 */
class IFrameProvider {
 public:
  virtual ~IFrameProvider() = default;

  /**
   * @brief Gets an allocated frame buffer
   * @return Shared pointer to allocated frame buffer
   */
  virtual std::shared_ptr<FrameBufferBase> get_allocated_frame() = 0;

  /**
   * @brief Called when a new frame is completed
   * @param frame The completed frame buffer
   */
  virtual void on_new_frame(std::shared_ptr<FrameBufferBase> frame) = 0;
};

/**
 * @brief Interface for packet copy strategies
 */
class IPacketCopyStrategy {
 public:
  virtual ~IPacketCopyStrategy() = default;

  /**
   * @brief Process a packet using this strategy
   * @param converter Reference to converter for accessing frame data
   * @param payload Payload data pointer
   * @param payload_size Size of payload data
   */
  virtual void process_packet(PacketsToFramesConverter& converter, uint8_t* payload,
                              size_t payload_size) = 0;

  /**
   * @brief Execute the copy operation
   * @param converter Reference to converter for accessing frame data
   */
  virtual void execute_copy(PacketsToFramesConverter& converter) = 0;

  /**
   * @brief Reset strategy state for new frame
   */
  virtual void reset_state() = 0;

  /**
   * @brief Get strategy type
   * @return Copy strategy type
   */
  virtual CopyStrategy get_strategy_type() const = 0;

  /**
   * @brief Check if strategy has accumulated data waiting to be copied
   * @return True if there's pending copy data, false otherwise
   */
  virtual bool has_pending_copy() const = 0;

  /**
   * @brief Set source memory location for packet data and cache copy kind
   * @param src_location Source memory storage type
   * @param dst_location Destination memory storage type
   */
  virtual void set_memory_locations(nvidia::gxf::MemoryStorageType src_location,
                                    nvidia::gxf::MemoryStorageType dst_location) = 0;
};

/**
 * @brief Contiguous packet copy strategy
 */
class ContiguousStrategy : public IPacketCopyStrategy {
 public:
  void process_packet(PacketsToFramesConverter& converter, uint8_t* payload,
                      size_t payload_size) override;

  void execute_copy(PacketsToFramesConverter& converter) override;
  void reset_state() override;
  CopyStrategy get_strategy_type() const override { return CopyStrategy::CONTIGUOUS; }
  bool has_pending_copy() const override { return accumulated_contiguous_size_ > 0; }
  void set_memory_locations(nvidia::gxf::MemoryStorageType src_location,
                            nvidia::gxf::MemoryStorageType dst_location) override {
    src_memory_location_ = src_location;
    copy_kind_ = MemoryCopyHelper::get_copy_kind(src_location, dst_location);
  }

 private:
  size_t accumulated_contiguous_size_ = 0;
  uint8_t* current_payload_start_ptr_ = nullptr;
  nvidia::gxf::MemoryStorageType src_memory_location_ = nvidia::gxf::MemoryStorageType::kDevice;
  cudaMemcpyKind copy_kind_ = cudaMemcpyDeviceToDevice;
};

/**
 * @brief Strided packet copy strategy
 */
class StridedStrategy : public IPacketCopyStrategy {
 public:
  explicit StridedStrategy(const StrideInfo& stride_info) : stride_info_(stride_info) {}

  void process_packet(PacketsToFramesConverter& converter, uint8_t* payload,
                      size_t payload_size) override;

  void execute_copy(PacketsToFramesConverter& converter) override;
  void reset_state() override;
  CopyStrategy get_strategy_type() const override { return CopyStrategy::STRIDED; }
  bool has_pending_copy() const override { return packet_count_ > 0; }
  void set_memory_locations(nvidia::gxf::MemoryStorageType src_location,
                            nvidia::gxf::MemoryStorageType dst_location) override {
    src_memory_location_ = src_location;
    copy_kind_ = MemoryCopyHelper::get_copy_kind(src_location, dst_location);
  }

 private:
  /**
   * @brief Execute accumulated strided copy and reset accumulation state
   * @param converter Reference to converter for accessing frame data
   */
  void execute_accumulated_strided_copy(PacketsToFramesConverter& converter);

  /**
   * @brief Execute individual packet copy (fallback when stride breaks)
   * @param converter Reference to converter for accessing frame data
   * @param payload_ptr Pointer to payload data
   * @param payload_size Size of payload data
   */
  void execute_individual_copy(PacketsToFramesConverter& converter, uint8_t* payload_ptr,
                               size_t payload_size);

  /**
   * @brief Check if current packet maintains expected stride pattern
   * @param payload Current packet payload pointer
   * @param payload_size Current packet payload size
   * @return True if stride is maintained, false if broken
   */
  bool is_stride_maintained(uint8_t* payload, size_t payload_size);

  /**
   * @brief Detect if buffer wrap-around occurred
   * @param current_ptr Current packet pointer
   * @param expected_ptr Expected pointer based on stride
   * @return True if wrap-around detected
   */
  bool detect_buffer_wraparound(uint8_t* current_ptr, uint8_t* expected_ptr) const;

  /**
   * @brief Reset accumulation state and start fresh with new packet
   * @param payload New packet payload pointer
   * @param payload_size New packet payload size
   */
  void reset_accumulation_state(uint8_t* payload, size_t payload_size);

 private:
  StrideInfo stride_info_;

  // Strided copy accumulation state
  uint8_t* first_packet_ptr_ = nullptr;
  uint8_t* last_packet_ptr_ = nullptr;
  size_t packet_count_ = 0;
  size_t total_data_size_ = 0;

  // Stride validation state
  bool stride_validated_ = false;
  size_t actual_stride_ = 0;

  nvidia::gxf::MemoryStorageType src_memory_location_ = nvidia::gxf::MemoryStorageType::kDevice;
  cudaMemcpyKind copy_kind_ = cudaMemcpyDeviceToDevice;

  // Buffer wrap-around detection threshold (1MB)
  static constexpr size_t WRAPAROUND_THRESHOLD = 1024 * 1024;
};

/**
 * @brief Strategy detector for analyzing packet patterns
 */
class PacketCopyStrategyDetector {
 public:
  static constexpr size_t STRATEGY_DETECTION_PACKETS = 4;  // Number of packets to analyze

  /**
   * @brief Configure detector with burst configuration parameters
   * @param header_stride_size Header stride size from burst info
   * @param payload_stride_size Payload stride size from burst info
   * @param hds_on Whether header data splitting is enabled
   */
  void configure_burst_parameters(size_t header_stride_size, size_t payload_stride_size,
                                  bool hds_on);

  /**
   * @brief Collect packet information for detection
   * @param rtp_params Parsed RTP parameters from header
   * @param payload Payload pointer for current packet
   * @param payload_size Size of payload data
   * @return True if enough packets collected for detection
   */
  bool collect_packet_info(const RtpParams& rtp_params, uint8_t* payload, size_t payload_size);

  /**
   * @brief Detect optimal copy strategy
   * @return Unique pointer to detected strategy, nullptr if detection failed
   */
  std::unique_ptr<IPacketCopyStrategy> detect_strategy();

  /**
   * @brief Check if strategy has been confirmed
   * @return True if strategy detection is complete
   */
  bool is_strategy_confirmed() const { return strategy_confirmed_; }

  /**
   * @brief Reset detector for new detection cycle
   */
  void reset_detection();

  /**
   * @brief Get number of packets analyzed
   * @return Number of packets analyzed
   */
  size_t get_packets_analyzed() const { return packets_analyzed_; }

 private:
  /**
   * @brief Analyze collected packet information to determine strategy
   * @return Detected strategy info, nullopt if detection failed
   */
  std::optional<std::pair<CopyStrategy, StrideInfo>> analyze_packet_pattern();

  /**
   * @brief Check if RTP sequences are consecutive (no drops)
   * @return True if no drops detected, false otherwise
   */
  bool validate_rtp_sequence_continuity() const;

  /**
   * @brief Detect cyclic buffer wrap-around in pointer sequence
   * @return True if wrap-around detected, false otherwise
   */
  bool detect_cyclic_buffer_wraparound() const;

 private:
  // Detection state
  std::vector<uint8_t*> detection_payloads_;
  std::vector<size_t> detection_payload_sizes_;
  std::vector<uint64_t> detection_rtp_sequences_;
  size_t packets_analyzed_ = 0;
  bool strategy_confirmed_ = false;

  // Burst configuration
  size_t expected_header_stride_ = 0;
  size_t expected_payload_stride_ = 0;
  bool expected_hds_on_ = false;
};

/**
 * @brief Converter for transforming packets to frames with optimized copy strategies
 */
class PacketsToFramesConverter {
 public:
  /**
   * @brief Constructor
   * @param frame_provider Shared pointer to frame provider interface
   */
  explicit PacketsToFramesConverter(std::shared_ptr<IFrameProvider> frame_provider);

  /**
   * @brief Factory method to create converter with raw @ref IFrameProvider pointer
   * @param provider Raw pointer to frame provider (must outlive the converter)
   * @return Unique pointer to the created converter
   * @note This creates a non-owning shared_ptr wrapper for compatibility
   */
  static std::unique_ptr<PacketsToFramesConverter> create(IFrameProvider* provider);

  /**
   * @brief Process an incoming packet
   * @param rtp_params Parsed RTP parameters from header
   * @param payload Payload data pointer
   */
  void process_incoming_packet(const RtpParams& rtp_params, uint8_t* payload);

  /**
   * @brief Configure with burst stride information (only affects detection if not already
   * confirmed)
   * @param header_stride_size Header stride size from burst info
   * @param payload_stride_size Payload stride size from burst info
   * @param hds_on Whether header data splitting is enabled
   */
  void configure_burst_parameters(size_t header_stride_size, size_t payload_stride_size,
                                  bool hds_on);

  /**
   * @brief Set source memory location for packet data
   * @param payload_on_cpu Whether payload data is on CPU (true) or GPU (false)
   */
  void set_source_memory_location(bool payload_on_cpu);

  /**
   * @brief Set source memory location for packet data using storage type
   * @param src_storage_type Source memory storage type
   */
  void set_source_memory_location(nvidia::gxf::MemoryStorageType src_storage_type);

  /**
   * @brief Reset converter state for new frame sequence
   */
  void reset_frame_state();

  /**
   * @brief Get current copy strategy (for debugging/monitoring)
   * @return Current copy strategy
   */
  CopyStrategy get_current_strategy() const;

  /**
   * @brief Force strategy re-detection (for testing or configuration changes)
   */
  void force_strategy_redetection();

  /**
   * @brief Check if there is accumulated packet data waiting to be copied
   * @return True if there is pending copy data, false otherwise
   */
  bool has_pending_copy() const;

  // Friend classes for accessing internal state
  friend class ContiguousStrategy;
  friend class StridedStrategy;

  /**
   * @brief Get destination frame buffer for copy operations
   * @return Shared pointer to current destination frame buffer
   */
  std::shared_ptr<FrameBufferBase> get_destination_frame_buffer() const { return frame_; }

  /**
   * @brief Get current byte position in frame
   * @return Current byte position
   */
  size_t get_frame_position() const { return current_byte_in_frame_; }

  /**
   * @brief Advance current byte position in frame
   * @param bytes Bytes to advance
   */
  void advance_frame_position(size_t bytes) { current_byte_in_frame_ += bytes; }

 private:
  /**
   * @brief Validate packet integrity and frame completion status
   * @param rtp_params Parsed RTP parameters from header
   * @return Pair of corruption status and error message
   */
  std::pair<bool, std::string> validate_packet_integrity(const RtpParams& rtp_params);

  /**
   * @brief Handle strategy detection phase for incoming packet
   * @param rtp_params Parsed RTP parameters from header
   * @param payload Payload data pointer
   * @param payload_size Size of payload data
   * @return True if packet should be processed, false if still detecting
   */
  bool handle_strategy_detection(const RtpParams& rtp_params, uint8_t* payload,
                                 size_t payload_size);

  /**
   * @brief Ensure strategy is available (create fallback if needed)
   */
  void ensure_strategy_available();

  /**
   * @brief Handle end of frame processing
   */
  void handle_end_of_frame();

 private:
  std::shared_ptr<IFrameProvider> frame_provider_;
  std::shared_ptr<FrameBufferBase> frame_;

  // Current frame state
  bool waiting_for_end_of_frame_;
  size_t current_byte_in_frame_;

  // Strategy components
  PacketCopyStrategyDetector detector_;
  std::unique_ptr<IPacketCopyStrategy> current_strategy_;

  nvidia::gxf::MemoryStorageType source_memory_location_ = nvidia::gxf::MemoryStorageType::kDevice;
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_PACKETS_TO_FRAMES_CONVERTER_H_
