/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_RX_FRAME_ASSEMBLY_CONTROLLER_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_RX_FRAME_ASSEMBLY_CONTROLLER_H_

#include <memory>
#include <string>
#include <functional>
#include "frame_provider.h"
#include "advanced_network/common.h"
#include "../common/adv_network_media_common.h"
#include "../common/frame_buffer.h"

namespace holoscan::ops {

namespace detail {

// Import public interfaces for cleaner usage
using holoscan::ops::IFrameProvider;

// Forward declaration - interface defined in memory_copy_strategies.h
class IMemoryCopyStrategy;

/**
 * @brief Main frame processing states
 */
enum class FrameState {
  IDLE,               // No active frame, awaiting first packet
  RECEIVING_PACKETS,  // Actively receiving and processing packets
  ERROR_RECOVERY      // Frame corrupted, waiting for recovery marker
};

/**
 * @brief State machine events that drive transitions
 */
enum class StateEvent {
  PACKET_ARRIVED,       // Regular packet received
  MARKER_DETECTED,      // M-bit packet received
  COPY_EXECUTED,        // Copy operation completed
  CORRUPTION_DETECTED,  // Frame corruption detected
  RECOVERY_MARKER,      // M-bit received during error recovery
  STRATEGY_DETECTED,    // Memory copy strategy detection completed
  FRAME_COMPLETED       // Frame processing finished
};

// Forward declaration - enum defined in memory_copy_strategies.h
enum class CopyStrategy;

/**
 * @brief Stride information for strided copy operations
 */
struct StrideInfo {
  size_t stride_size = 0;   // Stride between packet payloads
  size_t payload_size = 0;  // Size of each payload
};

/**
 * @brief Result of state machine transition
 */
struct StateTransitionResult {
  bool success = false;                           // Whether transition succeeded
  FrameState new_frame_state = FrameState::IDLE;  // New state after transition
  bool should_execute_copy = false;               // Whether copy operation should be executed
  bool should_complete_frame = false;             // Whether frame completion should be triggered
  bool should_emit_frame = false;                 // Whether frame should be emitted
  bool should_allocate_new_frame = false;         // Whether new frame should be allocated
  bool should_skip_memory_copy_processing =
      false;                  // Whether to skip memory copy processing (e.g., during recovery)
  std::string error_message;  // Error description if success=false
};

/**
 * @brief Assembly controller context (internal state)
 */
struct FrameAssemblyContext {
  FrameState frame_state = FrameState::IDLE;
  size_t frame_position = 0;                       // Current byte position in frame
  std::shared_ptr<FrameBufferBase> current_frame;  // Active frame buffer
};

/**
 * @brief Main frame processing state machine
 *
 * This class provides centralized state management for the entire packet-to-frame
 * conversion process, coordinating between strategies, frame allocation, and
 * error handling.
 *
 * @note Design Principle: This state machine focuses purely on state transitions
 *       and does not directly process packet data. All methods accept rtp_params
 *       and payload parameters for API consistency and future extensibility, but
 *       these parameters are currently unused. Actual packet processing is handled
 *       by the strategy layer (IMemoryCopyStrategy implementations).
 */
class FrameAssemblyController {
 public:
  /**
   * @brief Constructor
   * @param frame_provider Provider for frame allocation
   */
  explicit FrameAssemblyController(std::shared_ptr<IFrameProvider> frame_provider);

  /**
   * @brief Process state machine event
   * @param event Event to process
   * @param rtp_params RTP packet parameters (currently unused, reserved for future use)
   * @param payload Packet payload (currently unused, reserved for future use)
   * @return Transition result with actions to execute
   *
   * @note This state machine focuses purely on state transitions based on events.
   *       Packet data processing is handled by the strategy layer. The rtp_params
   *       and payload parameters are provided for API consistency and future
   *       extensibility but are not currently used in state transition logic.
   */
  StateTransitionResult process_event(StateEvent event, const RtpParams* rtp_params = nullptr,
                                      uint8_t* payload = nullptr);

  /**
   * @brief Reset assembly controller to initial state
   */
  void reset();

  /**
   * @brief Get current frame state
   * @return Current state of the assembly controller
   */
  FrameState get_frame_state() const { return context_.frame_state; }

  /**
   * @brief Get current frame buffer
   * @return Current frame or nullptr
   */
  std::shared_ptr<FrameBufferBase> get_current_frame() const { return context_.current_frame; }

  /**
   * @brief Get current frame position
   * @return Current byte position in frame
   */
  size_t get_frame_position() const { return context_.frame_position; }

  /**
   * @brief Advance frame position (with bounds checking)
   * @param bytes Number of bytes to advance
   * @return True if advancement succeeded
   */
  bool advance_frame_position(size_t bytes);

  /**
   * @brief Set strategy (for compatibility with old interface)
   * @param strategy Strategy to use (can be nullptr)
   */
  void set_strategy(std::shared_ptr<IMemoryCopyStrategy> strategy);

  /**
   * @brief Get currently set strategy
   * @return Current strategy or nullptr
   */
  std::shared_ptr<IMemoryCopyStrategy> get_strategy() const { return strategy_; }

  /**
   * @brief Allocate new frame for processing
   * @return True if allocation succeeded
   */
  bool allocate_new_frame();

  /**
   * @brief Release current frame back to the pool
   */
  void release_current_frame();

 private:
  /**
   * @brief Validate frame bounds for operations
   * @param required_bytes Number of bytes that will be written
   * @return True if operation is safe
   */
  bool validate_frame_bounds(size_t required_bytes) const;

  /**
   * @brief Transition to new state with validation
   * @param new_state Target state
   * @return True if transition is valid
   */
  bool transition_to_state(FrameState new_state);

  /**
   * @brief Handle IDLE state events
   * @param event Event to process
   * @param rtp_params RTP parameters
   * @param payload Packet payload
   * @return Transition result
   */
  StateTransitionResult handle_idle_state(StateEvent event, const RtpParams* rtp_params,
                                          uint8_t* payload);

  /**
   * @brief Handle RECEIVING_PACKETS state events
   * @param event Event to process
   * @param rtp_params RTP parameters
   * @param payload Packet payload
   * @return Transition result
   */
  StateTransitionResult handle_receiving_state(StateEvent event, const RtpParams* rtp_params,
                                               uint8_t* payload);

  /**
   * @brief Handle ERROR_RECOVERY state events
   * @param event Event to process
   * @param rtp_params RTP parameters
   * @param payload Packet payload
   * @return Transition result
   */
  StateTransitionResult handle_error_recovery_state(StateEvent event, const RtpParams* rtp_params,
                                                    uint8_t* payload);

  /**
   * @brief Create successful transition result
   * @param new_state New state after transition
   * @return Success result
   */
  StateTransitionResult create_success_result(FrameState new_state);

  /**
   * @brief Create error transition result
   * @param error_message Error description
   * @return Error result
   */
  StateTransitionResult create_error_result(const std::string& error_message);

 private:
  // Core components
  std::shared_ptr<IFrameProvider> frame_provider_;
  std::shared_ptr<IMemoryCopyStrategy> strategy_;

  // Assembly controller context
  FrameAssemblyContext context_;

  // Statistics and debugging
  size_t packets_processed_ = 0;
  size_t frames_completed_ = 0;
  size_t error_recoveries_ = 0;
};

/**
 * @brief Utility functions for frame assembly operations
 */
class FrameAssemblyHelper {
 public:
  /**
   * @brief Convert state to string for logging
   * @param state Frame state
   * @return String representation
   */
  static std::string state_to_string(FrameState state);

  /**
   * @brief Convert event to string for logging
   * @param event State event
   * @return String representation
   */
  static std::string event_to_string(StateEvent event);

  /**
   * @brief Check if state transition is valid
   * @param from_state Source state
   * @param to_state Target state
   * @return True if transition is allowed
   */
  static bool is_valid_transition(FrameState from_state, FrameState to_state);

  /**
   * @brief Get expected events for a given state
   * @param state Current state
   * @return List of valid events for the state
   */
  static std::vector<StateEvent> get_valid_events(FrameState state);
};

}  // namespace detail

}  // namespace holoscan::ops

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_RX_FRAME_ASSEMBLY_CONTROLLER_H_
