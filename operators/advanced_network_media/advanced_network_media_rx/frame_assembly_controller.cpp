/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "frame_assembly_controller.h"
#include "memory_copy_strategies.h"
#include "../common/adv_network_media_common.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <holoscan/logger/logger.hpp>

namespace holoscan::ops {
namespace detail {

// Import public interfaces for cleaner usage
using holoscan::ops::IFrameProvider;

// ========================================================================================
// FrameAssemblyController Implementation
// ========================================================================================

FrameAssemblyController::FrameAssemblyController(std::shared_ptr<IFrameProvider> frame_provider)
    : frame_provider_(frame_provider) {
  if (!frame_provider_) {
    throw std::invalid_argument("FrameProvider cannot be null");
  }

  // Don't allocate frame in constructor - wait for first packet
  // This prevents reducing the pool size unnecessarily
  context_.current_frame = nullptr;
  context_.frame_position = 0;
  context_.frame_state = FrameState::IDLE;

  ANM_CONFIG_LOG("FrameAssemblyController initialized");
}

StateTransitionResult FrameAssemblyController::process_event(StateEvent event,
                                                             const RtpParams* rtp_params,
                                                             uint8_t* payload) {
  // NOTE: rtp_params and payload are currently unused in state transition logic.
  // This assembly controller focuses purely on event-driven state transitions.
  // Parameters are kept for API consistency and future extensibility.
  (void)rtp_params;  // Suppress unused parameter warning
  (void)payload;     // Suppress unused parameter warning

  packets_processed_++;

  ANM_STATE_TRACE(
      "State machine context: frame_position={}, current_frame={}, packets_processed={}",
      context_.frame_position,
      context_.current_frame ? "allocated" : "null",
      packets_processed_);

  StateTransitionResult result;

  // Route to appropriate state handler
  ANM_STATE_TRACE("Routing event {} to state handler for state {}",
                  FrameAssemblyHelper::event_to_string(event),
                  FrameAssemblyHelper::state_to_string(context_.frame_state));

  switch (context_.frame_state) {
    case FrameState::IDLE:
      result = handle_idle_state(event, rtp_params, payload);
      break;
    case FrameState::RECEIVING_PACKETS:
      result = handle_receiving_state(event, rtp_params, payload);
      break;
    case FrameState::ERROR_RECOVERY:
      result = handle_error_recovery_state(event, rtp_params, payload);
      break;
    default:
      ANM_STATE_TRACE("Unhandled state encountered: {}",
                      FrameAssemblyHelper::state_to_string(context_.frame_state));
      result = create_error_result("Unknown state");
      break;
  }

  // Attempt state transition whenever new_frame_state differs from current state
  if (result.new_frame_state != context_.frame_state) {
    FrameState old_state = context_.frame_state;
    ANM_STATE_TRACE(
        "Attempting state transition: {} -> {} with result actions: allocate={}, complete={}, "
        "emit={}",
        FrameAssemblyHelper::state_to_string(old_state),
        FrameAssemblyHelper::state_to_string(result.new_frame_state),
        result.should_allocate_new_frame,
        result.should_complete_frame,
        result.should_emit_frame);

    if (transition_to_state(result.new_frame_state)) {
      ANM_STATE_TRACE("State transition: {} -> {}",
                      FrameAssemblyHelper::state_to_string(old_state),
                      FrameAssemblyHelper::state_to_string(result.new_frame_state));
    } else {
      result = create_error_result("Invalid state transition");
    }
  } else if (result.success) {
    ANM_STATE_TRACE(
        "Event processed, staying in state {} with result actions: allocate={}, complete={}, "
        "emit={}",
        FrameAssemblyHelper::state_to_string(context_.frame_state),
        result.should_allocate_new_frame,
        result.should_complete_frame,
        result.should_emit_frame);
  }

  return result;
}

void FrameAssemblyController::reset() {
  context_.frame_state = FrameState::IDLE;
  context_.frame_position = 0;

  // Reset memory copy strategy if set
  if (strategy_) {
    strategy_->reset();
  }

  // Don't allocate frame in reset - wait for first packet
  // This prevents reducing the pool size unnecessarily
  context_.current_frame = nullptr;

  ANM_CONFIG_LOG("Assembly controller reset to initial state");
}

bool FrameAssemblyController::advance_frame_position(size_t bytes) {
  if (!validate_frame_bounds(bytes)) {
    ANM_LOG_ERROR(
        "Frame position advancement would exceed bounds: current={}, bytes={}, frame_size={}",
        context_.frame_position,
        bytes,
        context_.current_frame ? context_.current_frame->get_size() : 0);
    return false;
  }

  context_.frame_position += bytes;

  ANM_FRAME_TRACE(
      "Frame position advanced by {} bytes to position {}", bytes, context_.frame_position);
  return true;
}

void FrameAssemblyController::set_strategy(std::shared_ptr<IMemoryCopyStrategy> strategy) {
  strategy_ = strategy;

  if (strategy_) {
    ANM_CONFIG_LOG("Strategy set: {}",
                   strategy_->get_type() == CopyStrategy::CONTIGUOUS ? "CONTIGUOUS" : "STRIDED");
  } else {
    ANM_CONFIG_LOG("Strategy cleared");
  }
}

bool FrameAssemblyController::allocate_new_frame() {
  context_.current_frame = frame_provider_->get_new_frame();
  context_.frame_position = 0;

  if (!context_.current_frame) {
    ANM_LOG_ERROR("Frame allocation failed");
    return false;
  }

  ANM_FRAME_TRACE("New frame allocated: size={}", context_.current_frame->get_size());
  return true;
}

void FrameAssemblyController::release_current_frame() {
  if (context_.current_frame) {
    ANM_FRAME_TRACE("Releasing current frame back to pool: size={}",
                    context_.current_frame->get_size());
    // Return frame to pool through frame provider
    frame_provider_->return_frame_to_pool(context_.current_frame);
    context_.current_frame.reset();
    context_.frame_position = 0;
  }
}

bool FrameAssemblyController::validate_frame_bounds(size_t required_bytes) const {
  if (!context_.current_frame) {
    return false;
  }

  return (context_.frame_position + required_bytes <= context_.current_frame->get_size());
}

bool FrameAssemblyController::transition_to_state(FrameState new_state) {
  if (!FrameAssemblyHelper::is_valid_transition(context_.frame_state, new_state)) {
    ANM_LOG_ERROR("Invalid state transition: {} -> {}",
                  FrameAssemblyHelper::state_to_string(context_.frame_state),
                  FrameAssemblyHelper::state_to_string(new_state));
    return false;
  }

  context_.frame_state = new_state;
  return true;
}

StateTransitionResult FrameAssemblyController::handle_idle_state(StateEvent event,
                                                                 const RtpParams* rtp_params,
                                                                 uint8_t* payload) {
  ANM_STATE_TRACE("IDLE state handler: processing event {}, current_frame={}",
                  FrameAssemblyHelper::event_to_string(event),
                  context_.current_frame ? "allocated" : "null");

  switch (event) {
    case StateEvent::PACKET_ARRIVED: {
      // Start receiving packets - allocate frame if we don't have one
      auto result = create_success_result(FrameState::RECEIVING_PACKETS);
      if (!context_.current_frame) {
        result.should_allocate_new_frame = true;
      }
      return result;
    }

    case StateEvent::MARKER_DETECTED: {
      // Single packet frame (edge case) - complete atomically
      auto result = create_success_result(FrameState::IDLE);

      if (!context_.current_frame) {
        // No frame allocated yet - allocate one for this single packet
        ANM_STATE_TRACE(
            "IDLE single-packet frame: no frame allocated, requesting new frame for single packet");
        result.should_allocate_new_frame = true;
        // Don't complete/emit since we just allocated
        return result;
      } else {
        // Frame exists - complete it and allocate new one
        ANM_STATE_TRACE(
            "IDLE single-packet frame: completing existing frame and requesting new one");
        result.should_complete_frame = true;
        result.should_emit_frame = true;
        if (frame_provider_->has_available_frames()) {
          result.should_allocate_new_frame = true;
        } else {
          ANM_LOG_WARN("Frame completed but pool is empty - staying in IDLE without new frame");
        }
        frames_completed_++;
        return result;
      }
    }

    case StateEvent::CORRUPTION_DETECTED:
      return create_success_result(FrameState::ERROR_RECOVERY);

    case StateEvent::STRATEGY_DETECTED:
      // This should not happen in IDLE state - memory copy strategy detection requires packets
      ANM_LOG_WARN("STRATEGY_DETECTED event received in IDLE state - treating as PACKET_ARRIVED");
      return create_success_result(FrameState::RECEIVING_PACKETS);

    default:
      return create_error_result("Unexpected event in IDLE state");
  }
}

StateTransitionResult FrameAssemblyController::handle_receiving_state(StateEvent event,
                                                                      const RtpParams* rtp_params,
                                                                      uint8_t* payload) {
  ANM_STATE_TRACE("RECEIVING_PACKETS state handler: processing event {}, frame_position={}",
                  FrameAssemblyHelper::event_to_string(event),
                  context_.frame_position);

  switch (event) {
    case StateEvent::PACKET_ARRIVED:
      // Continue receiving packets
      return create_success_result(FrameState::RECEIVING_PACKETS);

    case StateEvent::MARKER_DETECTED: {
      // Frame completion triggered - complete atomically
      // Only allocate new frame if pool has available frames
      ANM_STATE_TRACE("RECEIVING_PACKETS->IDLE: marker detected, completing frame at position {}",
                      context_.frame_position);
      auto result = create_success_result(FrameState::IDLE);
      result.should_complete_frame = true;
      result.should_emit_frame = true;
      if (frame_provider_->has_available_frames()) {
        result.should_allocate_new_frame = true;
      } else {
        ANM_LOG_WARN("Frame completed but pool is empty - staying in IDLE without new frame");
      }
      frames_completed_++;
      return result;
    }

    case StateEvent::COPY_EXECUTED:
      // This should not happen - COPY_EXECUTED events are not sent to state machine
      ANM_LOG_WARN("COPY_EXECUTED received in RECEIVING_PACKETS state - this indicates dead code");
      return create_success_result(FrameState::RECEIVING_PACKETS);

    case StateEvent::CORRUPTION_DETECTED:
      return create_success_result(FrameState::ERROR_RECOVERY);

    case StateEvent::STRATEGY_DETECTED:
      // Memory copy strategy detection completed while receiving packets
      return create_success_result(FrameState::RECEIVING_PACKETS);

    default:
      return create_error_result("Unexpected event in RECEIVING_PACKETS state");
  }
}

StateTransitionResult FrameAssemblyController::handle_error_recovery_state(
    StateEvent event, const RtpParams* rtp_params, uint8_t* payload) {
  ANM_STATE_TRACE(
      "ERROR_RECOVERY state handler: processing event {}, current_frame={}, error_recoveries={}",
      FrameAssemblyHelper::event_to_string(event),
      context_.current_frame ? "allocated" : "null",
      error_recoveries_);

  switch (event) {
    case StateEvent::RECOVERY_MARKER: {
      // Recovery marker received - release any corrupted frame and try to start new one
      // Release corrupted frame first to free the pool slot, then check availability
      if (context_.current_frame) {
        ANM_FRAME_TRACE("Releasing corrupted frame during recovery: size={}, ptr={}",
                        context_.current_frame->get_size(),
                        static_cast<void*>(context_.current_frame->get()));
        ANM_STATE_TRACE("ERROR_RECOVERY: releasing corrupted frame before recovery completion");
        release_current_frame();
      }

      // Only exit recovery if we can allocate a new frame
      if (frame_provider_->has_available_frames()) {
        error_recoveries_++;
        ANM_LOG_INFO("Recovery marker (M-bit) received - exiting error recovery");
        ANM_STATE_TRACE(
            "ERROR_RECOVERY->IDLE: recovery successful, requesting new frame, total recoveries={}",
            error_recoveries_);
        auto result = create_success_result(FrameState::IDLE);
        result.should_allocate_new_frame = true;
        return result;
      } else {
        // Stay in recovery if no frames available
        ANM_LOG_WARN("Recovery marker detected but frame pool is empty - staying in recovery");
        auto result = create_success_result(FrameState::ERROR_RECOVERY);
        return result;
      }
    }

    case StateEvent::PACKET_ARRIVED: {
      // Stay in recovery state, waiting for marker - packet discarded
      ANM_STATE_TRACE("ERROR_RECOVERY: packet discarded, packets_processed={}", packets_processed_);
      auto result = create_success_result(FrameState::ERROR_RECOVERY);
      result.should_skip_memory_copy_processing = true;
      return result;
    }

    case StateEvent::CORRUPTION_DETECTED:
      // Additional corruption detected, stay in recovery
      return create_success_result(FrameState::ERROR_RECOVERY);

    case StateEvent::MARKER_DETECTED: {
      // This should not happen in ERROR_RECOVERY - should be RECOVERY_MARKER instead
      ANM_LOG_WARN(
          "MARKER_DETECTED received in ERROR_RECOVERY state - treating as RECOVERY_MARKER");
      // Release corrupted frame first to free the pool slot, then check availability
      if (context_.current_frame) {
        ANM_FRAME_TRACE("Releasing corrupted frame during recovery: size={}, ptr={}",
                        context_.current_frame->get_size(),
                        static_cast<void*>(context_.current_frame->get()));
        ANM_STATE_TRACE("ERROR_RECOVERY: releasing corrupted frame before recovery completion");
        release_current_frame();
      }

      // Only exit recovery if we can allocate a new frame
      if (frame_provider_->has_available_frames()) {
        error_recoveries_++;
        auto result = create_success_result(FrameState::IDLE);
        result.should_allocate_new_frame = true;
        return result;
      } else {
        // Stay in recovery if no frames available
        ANM_LOG_WARN("Marker detected but frame pool is empty - staying in recovery");
        auto result = create_success_result(FrameState::ERROR_RECOVERY);
        return result;
      }
    }

    default:
      return create_error_result("Unexpected event in ERROR_RECOVERY state");
  }
}

StateTransitionResult FrameAssemblyController::create_success_result(FrameState new_state) {
  StateTransitionResult result;
  result.success = true;
  result.new_frame_state = new_state;
  return result;
}

StateTransitionResult FrameAssemblyController::create_error_result(
    const std::string& error_message) {
  StateTransitionResult result;
  result.success = false;
  result.error_message = error_message;
  result.new_frame_state = FrameState::ERROR_RECOVERY;
  return result;
}

// ========================================================================================
// FrameAssemblyHelper Implementation
// ========================================================================================

std::string FrameAssemblyHelper::state_to_string(FrameState state) {
  switch (state) {
    case FrameState::IDLE:
      return "IDLE";
    case FrameState::RECEIVING_PACKETS:
      return "RECEIVING_PACKETS";
    case FrameState::ERROR_RECOVERY:
      return "ERROR_RECOVERY";
    default:
      return "UNKNOWN";
  }
}

std::string FrameAssemblyHelper::event_to_string(StateEvent event) {
  switch (event) {
    case StateEvent::PACKET_ARRIVED:
      return "PACKET_ARRIVED";
    case StateEvent::MARKER_DETECTED:
      return "MARKER_DETECTED";
    case StateEvent::COPY_EXECUTED:
      return "COPY_EXECUTED";
    case StateEvent::CORRUPTION_DETECTED:
      return "CORRUPTION_DETECTED";
    case StateEvent::RECOVERY_MARKER:
      return "RECOVERY_MARKER";
    case StateEvent::STRATEGY_DETECTED:
      return "STRATEGY_DETECTED";
    case StateEvent::FRAME_COMPLETED:
      return "FRAME_COMPLETED";
    default:
      return "UNKNOWN";
  }
}

bool FrameAssemblyHelper::is_valid_transition(FrameState from_state, FrameState to_state) {
  // Define valid state transitions
  switch (from_state) {
    case FrameState::IDLE:
      return (to_state == FrameState::RECEIVING_PACKETS || to_state == FrameState::ERROR_RECOVERY);

    case FrameState::RECEIVING_PACKETS:
      return (to_state == FrameState::RECEIVING_PACKETS || to_state == FrameState::IDLE ||
              to_state == FrameState::ERROR_RECOVERY);

    case FrameState::ERROR_RECOVERY:
      return (to_state == FrameState::IDLE || to_state == FrameState::ERROR_RECOVERY);

    default:
      return false;
  }
}

std::vector<StateEvent> FrameAssemblyHelper::get_valid_events(FrameState state) {
  switch (state) {
    case FrameState::IDLE:
      return {
          StateEvent::PACKET_ARRIVED, StateEvent::MARKER_DETECTED, StateEvent::CORRUPTION_DETECTED};

    case FrameState::RECEIVING_PACKETS:
      return {StateEvent::PACKET_ARRIVED,
              StateEvent::MARKER_DETECTED,
              StateEvent::CORRUPTION_DETECTED,
              StateEvent::STRATEGY_DETECTED};

    case FrameState::ERROR_RECOVERY:
      return {
          StateEvent::RECOVERY_MARKER, StateEvent::PACKET_ARRIVED, StateEvent::CORRUPTION_DETECTED};

    default:
      return {};
  }
}

}  // namespace detail
}  // namespace holoscan::ops
