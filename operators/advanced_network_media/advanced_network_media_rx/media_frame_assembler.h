/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_MEDIA_FRAME_ASSEMBLER_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_MEDIA_FRAME_ASSEMBLER_H_

#include <memory>
#include <functional>
#include <chrono>
#include <string>
#include "frame_provider.h"
#include "frame_assembly_controller.h"
#include "memory_copy_strategies.h"
#include "advanced_network/common.h"
#include "../common/frame_buffer.h"

namespace holoscan::ops {

// Forward declarations for detail namespace types used internally
namespace detail {
enum class StateEvent;
struct StateTransitionResult;
class FrameAssemblyController;
class MemoryCopyStrategyDetector;
class IMemoryCopyStrategy;
}  // namespace detail

// Import detail types for cleaner private method signatures
using detail::CopyStrategy;
using detail::FrameAssemblyController;
using detail::IMemoryCopyStrategy;
using detail::MemoryCopyStrategyDetector;
using detail::StateEvent;
using detail::StateTransitionResult;

/**
 * @brief Configuration for memory copy strategy detection and memory settings
 */
struct AssemblerConfiguration {
  // Memory configuration
  nvidia::gxf::MemoryStorageType source_memory_type = nvidia::gxf::MemoryStorageType::kDevice;
  nvidia::gxf::MemoryStorageType destination_memory_type = nvidia::gxf::MemoryStorageType::kDevice;

  // Burst configuration
  size_t header_stride_size = 0;
  size_t payload_stride_size = 0;
  bool hds_enabled = false;

  // Detection configuration
  bool force_contiguous_memory_copy_strategy = false;
  bool enable_memory_copy_strategy_detection = true;
};

/**
 * @brief Callback interface for frame completion events
 */
class IFrameCompletionHandler {
 public:
  virtual ~IFrameCompletionHandler() = default;

  /**
   * @brief Called when a frame is completed and ready for emission
   * @param frame Completed frame buffer
   */
  virtual void on_frame_completed(std::shared_ptr<FrameBufferBase> frame) = 0;

  /**
   * @brief Called when frame processing encounters an error
   * @param error_message Error description
   */
  virtual void on_frame_error(const std::string& error_message) = 0;
};

/**
 * @brief Frame assembler for converting packets to frames
 *
 * This class provides a clean, assembly controller driven approach to converting
 * network packets into video frames with automatic memory copy strategy detection and
 * robust error handling.
 *
 * @note Architecture: This class coordinates between three main components:
 *       - FrameAssemblyController: Assembly controller for state transitions
 *       - IMemoryCopyStrategy: Strategy pattern for packet data processing
 *       - MemoryCopyStrategyDetector: Automatic detection of optimal copy strategies
 *
 *       The assembly controller layer focuses purely on state management and does
 *       not directly process packet data, maintaining clean separation of concerns.
 */
class MediaFrameAssembler {
 public:
  /**
   * @brief Constructor
   * @param frame_provider Provider for frame allocation
   * @param config Assembler configuration
   */
  MediaFrameAssembler(std::shared_ptr<IFrameProvider> frame_provider,
                      const AssemblerConfiguration& config = {});

  /**
   * @brief Set frame completion handler
   * @param handler Callback handler for frame events
   */
  void set_completion_handler(std::shared_ptr<IFrameCompletionHandler> handler);

  /**
   * @brief Configure burst parameters for memory copy strategy detection
   * @param header_stride_size Header stride from burst info
   * @param payload_stride_size Payload stride from burst info
   * @param hds_enabled Whether header data split is enabled
   */
  void configure_burst_parameters(size_t header_stride_size, size_t payload_stride_size,
                                  bool hds_enabled);

  /**
   * @brief Update memory configuration
   * @param source_type Source memory storage type
   * @param destination_type Destination memory storage type
   */
  void configure_memory_types(nvidia::gxf::MemoryStorageType source_type,
                              nvidia::gxf::MemoryStorageType destination_type);

  /**
   * @brief Process incoming RTP packet - MAIN ENTRY POINT
   * @param rtp_params Parsed RTP parameters
   * @param payload Packet payload data
   */
  void process_incoming_packet(const RtpParams& rtp_params, uint8_t* payload);

  /**
   * @brief Force memory copy strategy redetection (for testing or config changes)
   */
  void force_memory_copy_strategy_redetection();

  /**
   * @brief Reset Media Frame Assembler to initial state
   */
  void reset();

  /**
   * @brief Get current Media Frame Assembler statistics
   * @return Statistics structure
   */
  struct Statistics {
    // Basic counters
    size_t packets_processed = 0;
    size_t frames_completed = 0;
    size_t errors_recovered = 0;
    size_t memory_copy_strategy_redetections = 0;

    // Enhanced frame tracking
    size_t current_frame_number = 0;           // Current frame being assembled
    size_t frames_started = 0;                 // Total frames started (including dropped)
    size_t frames_dropped = 0;                 // Frames dropped due to errors
    size_t frames_completed_successfully = 0;  // Successfully completed frames

    // Enhanced error tracking
    size_t sequence_discontinuities = 0;  // RTP sequence discontinuities
    size_t buffer_overflow_errors = 0;    // Buffer wraparound detections
    size_t memory_corruption_errors = 0;  // Memory bounds/corruption errors
    size_t error_recovery_cycles = 0;     // Number of error recovery cycles

    // Current frame metrics
    size_t packets_in_current_frame = 0;   // Packets accumulated in current frame
    size_t bytes_in_current_frame = 0;     // Bytes accumulated in current frame
    uint32_t last_sequence_number = 0;     // Last processed sequence number
    uint32_t first_sequence_in_frame = 0;  // First sequence number in current frame

    // State information
    std::string current_strategy = "UNKNOWN";
    std::string current_frame_state = "IDLE";
    std::string last_error;

    // Timing information (for frame rates/debugging)
#if ENABLE_STATISTICS_LOGGING
    std::chrono::steady_clock::time_point last_frame_completion_time;
    std::chrono::steady_clock::time_point last_error_time;
#endif
  };

  Statistics get_statistics() const;

  /**
   * @brief Get comprehensive statistics summary for debugging
   * @return Formatted string with detailed statistics
   */
  std::string get_statistics_summary() const;

  /**
   * @brief Check if Media Frame Assembler has accumulated data waiting to be copied
   * @return True if copy operations have accumulated data pending
   */
  bool has_accumulated_data() const;

  /**
   * @brief Get current frame for external operations (debugging)
   * @return Current frame buffer or nullptr
   */
  std::shared_ptr<FrameBufferBase> get_current_frame() const;

  /**
   * @brief Get current frame position (debugging)
   * @return Current byte position in frame
   */
  size_t get_frame_position() const;

 private:
  /**
   * @brief Determine assembly controller event from packet parameters
   * @param rtp_params RTP packet parameters
   * @param payload Packet payload
   * @return Appropriate state event
   */
  StateEvent determine_event(const RtpParams& rtp_params, uint8_t* payload);

  /**
   * @brief Execute actions based on assembly controller transition result
   * @param result State transition result
   * @param rtp_params RTP packet parameters
   * @param payload Packet payload
   */
  void execute_actions(const StateTransitionResult& result, const RtpParams& rtp_params,
                       uint8_t* payload);

  /**
   * @brief Handle memory copy strategy detection and setup
   * @param rtp_params RTP packet parameters
   * @param payload Packet payload
   * @return True if memory copy strategy is ready for processing
   */
  bool handle_memory_copy_strategy_detection(const RtpParams& rtp_params, uint8_t* payload);

  /**
   * @brief Set up memory copy strategy once detection is complete
   * @param strategy Detected memory copy strategy
   */
  void setup_memory_copy_strategy(std::unique_ptr<IMemoryCopyStrategy> strategy);

  /**
   * @brief Validate packet integrity
   * @param rtp_params RTP packet parameters
   * @return True if packet is valid
   */
  bool validate_packet_integrity(const RtpParams& rtp_params);

  /**
   * @brief Handle frame completion processing
   */
  void handle_frame_completion();

  /**
   * @brief Handle error recovery
   * @param error_message Error description
   */
  void handle_error_recovery(const std::string& error_message);

  /**
   * @brief Update statistics
   * @param event State event that occurred
   */
  void update_statistics(StateEvent event);

  /**
   * @brief Update packet-specific statistics
   * @param rtp_params RTP parameters from the packet
   */
  void update_packet_statistics(const RtpParams& rtp_params);

 private:
  // Core components
  std::unique_ptr<FrameAssemblyController> assembly_controller_;
  std::unique_ptr<MemoryCopyStrategyDetector> memory_copy_strategy_detector_;
  std::unique_ptr<IMemoryCopyStrategy> current_copy_strategy_;

  // Configuration
  AssemblerConfiguration config_;

  // Callback handlers
  std::shared_ptr<IFrameCompletionHandler> completion_handler_;

  // Statistics
  mutable Statistics statistics_;

  // State tracking
  bool memory_copy_strategy_detection_active_ = false;
};

/**
 * @brief Default frame completion handler that can be used with the Media Frame Assembler
 */
class DefaultFrameCompletionHandler : public IFrameCompletionHandler {
 public:
  /**
   * @brief Constructor
   * @param frame_ready_callback Callback for completed frames
   * @param error_callback Callback for errors
   */
  DefaultFrameCompletionHandler(
      std::function<void(std::shared_ptr<FrameBufferBase>)> frame_ready_callback,
      std::function<void(const std::string&)> error_callback = nullptr);

  // IFrameCompletionHandler interface
  void on_frame_completed(std::shared_ptr<FrameBufferBase> frame) override;
  void on_frame_error(const std::string& error_message) override;

 private:
  std::function<void(std::shared_ptr<FrameBufferBase>)> frame_ready_callback_;
  std::function<void(const std::string&)> error_callback_;
};

/**
 * @brief Utility functions for assembler configuration
 */
class AssemblerConfigurationHelper {
 public:
  /**
   * @brief Create configuration with burst parameters
   * @param header_stride Header stride size
   * @param payload_stride Payload stride size
   * @param hds_enabled HDS setting
   * @param payload_on_cpu Whether payload is in CPU memory
   * @param frames_on_host Whether frames should be in host memory
   * @return Assembler configuration
   */
  static AssemblerConfiguration create_with_burst_parameters(size_t header_stride,
                                                             size_t payload_stride,
                                                             bool hds_enabled, bool payload_on_cpu,
                                                             bool frames_on_host);

  /**
   * @brief Create configuration for testing with forced memory copy strategy
   * @param force_contiguous Whether to force contiguous memory copy strategy
   * @return Test configuration
   */
  static AssemblerConfiguration create_test_config(bool force_contiguous = true);

  /**
   * @brief Validate configuration parameters
   * @param config Configuration to validate
   * @return True if configuration is valid
   */
  static bool validate_configuration(const AssemblerConfiguration& config);
};

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_MEDIA_FRAME_ASSEMBLER_H_
