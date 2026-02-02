/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "media_frame_assembler.h"
#include "../common/adv_network_media_common.h"
#include <sstream>
#include <iomanip>
#include <string>
#include <memory>

namespace holoscan::ops {

// Import detail namespace classes for convenience
using detail::CopyStrategy;
using detail::FrameAssemblyController;
using detail::FrameState;
using detail::IMemoryCopyStrategy;
using detail::MemoryCopyStrategyDetector;
using detail::StateEvent;
using detail::StrategyFactory;

// RTP sequence number gap threshold for detecting potential buffer wraparound
constexpr int32_t kRtpSequenceGapWraparoundThreshold = 16384;  // 2^14

// Helper functions to convert internal types to strings for statistics
std::string convert_strategy_to_string(CopyStrategy internal_strategy) {
  switch (internal_strategy) {
    case CopyStrategy::CONTIGUOUS:
      return "CONTIGUOUS";
    case CopyStrategy::STRIDED:
      return "STRIDED";
    default:
      return "UNKNOWN";
  }
}

std::string convert_state_to_string(FrameState internal_state) {
  switch (internal_state) {
    case FrameState::IDLE:
      return "IDLE";
    case FrameState::RECEIVING_PACKETS:
      return "RECEIVING_PACKETS";

    case FrameState::ERROR_RECOVERY:
      return "ERROR_RECOVERY";
    default:
      return "IDLE";
  }
}

// ========================================================================================
// MediaFrameAssembler Implementation
// ========================================================================================

MediaFrameAssembler::MediaFrameAssembler(std::shared_ptr<IFrameProvider> frame_provider,
                                         const AssemblerConfiguration& config)
    : config_(config) {
  // Validate configuration
  if (!AssemblerConfigurationHelper::validate_configuration(config_)) {
    throw std::invalid_argument("Invalid assembler configuration");
  }

  // Create frame assembly controller
  assembly_controller_ = std::make_unique<FrameAssemblyController>(frame_provider);

  // Create memory copy strategy detector if needed
  if (config_.enable_memory_copy_strategy_detection &&
      !config_.force_contiguous_memory_copy_strategy) {
    memory_copy_strategy_detector_ = StrategyFactory::create_detector();
    memory_copy_strategy_detection_active_ = true;
  } else if (config_.force_contiguous_memory_copy_strategy) {
    // Create contiguous memory copy strategy immediately
    current_copy_strategy_ = StrategyFactory::create_contiguous_strategy(
        config_.source_memory_type, config_.destination_memory_type);
    setup_memory_copy_strategy(std::move(current_copy_strategy_));
    memory_copy_strategy_detection_active_ = false;
  }

  ANM_CONFIG_LOG("MediaFrameAssembler initialized: strategy_detection={}, force_contiguous={}",
                 config_.enable_memory_copy_strategy_detection,
                 config_.force_contiguous_memory_copy_strategy);
}

void MediaFrameAssembler::set_completion_handler(std::shared_ptr<IFrameCompletionHandler> handler) {
  completion_handler_ = handler;
}

void MediaFrameAssembler::configure_burst_parameters(size_t header_stride_size,
                                                     size_t payload_stride_size, bool hds_enabled) {
  config_.header_stride_size = header_stride_size;
  config_.payload_stride_size = payload_stride_size;
  config_.hds_enabled = hds_enabled;

  // Configure memory copy strategy detector if active
  if (memory_copy_strategy_detector_) {
    memory_copy_strategy_detector_->configure_burst_parameters(
        header_stride_size, payload_stride_size, hds_enabled);
  }

  ANM_CONFIG_LOG("Burst parameters configured: header_stride={}, payload_stride={}, hds={}",
                 header_stride_size,
                 payload_stride_size,
                 hds_enabled);
}

void MediaFrameAssembler::configure_memory_types(nvidia::gxf::MemoryStorageType source_type,
                                                 nvidia::gxf::MemoryStorageType destination_type) {
  config_.source_memory_type = source_type;
  config_.destination_memory_type = destination_type;

  ANM_CONFIG_LOG("Memory types configured: source={}, destination={}",
                 static_cast<int>(source_type),
                 static_cast<int>(destination_type));

  // If memory copy strategy is already set up, we may need to recreate it with new memory types
  if (current_copy_strategy_ && !memory_copy_strategy_detection_active_) {
    CopyStrategy strategy_type = current_copy_strategy_->get_type();

    if (strategy_type == CopyStrategy::CONTIGUOUS) {
      current_copy_strategy_ =
          StrategyFactory::create_contiguous_strategy(source_type, destination_type);
    }
    // Note: For strided memory copy strategy, we would need the stride info, so we'd trigger
    // redetection

    setup_memory_copy_strategy(std::move(current_copy_strategy_));
  }
}

void MediaFrameAssembler::process_incoming_packet(const RtpParams& rtp_params, uint8_t* payload) {
  try {
    // Update statistics
    update_statistics(StateEvent::PACKET_ARRIVED);
    update_packet_statistics(rtp_params);

    // Determine appropriate event for this packet
    StateEvent event = determine_event(rtp_params, payload);

    // Check current state before processing for recovery completion detection
    FrameState previous_state = assembly_controller_->get_frame_state();

    // Process event through assembly controller
    auto result = assembly_controller_->process_event(event, &rtp_params, payload);

    if (!result.success) {
      ANM_FRAME_ERROR(statistics_.current_frame_number,
                      "Assembly controller processing failed: {}",
                      result.error_message);
      handle_error_recovery(result.error_message);
      return;
    }

    // Log error recovery state changes
    if (result.new_frame_state == FrameState::ERROR_RECOVERY) {
      ANM_STATE_LOG("Error recovery active - discarding packets until M-bit marker received");
    } else if (previous_state == FrameState::ERROR_RECOVERY &&
               result.new_frame_state == FrameState::IDLE) {
      ANM_STATE_LOG("Error recovery completed successfully - resuming normal frame processing");
    }

    // Execute actions based on assembly controller result
    execute_actions(result, rtp_params, payload);

    ANM_PACKET_TRACE("Packet processed successfully: seq={}, event={}, new_state={}",
                     rtp_params.sequence_number,
                     static_cast<int>(event),
                     static_cast<int>(result.new_frame_state));

    // Special logging for recovery marker processing
    if (event == StateEvent::RECOVERY_MARKER) {
      ANM_STATE_LOG("RECOVERY_MARKER event processed - should have exited error recovery");
    }
  } catch (const std::exception& e) {
    std::string error_msg = std::string("Exception in packet processing: ") + e.what();
    ANM_FRAME_ERROR(statistics_.current_frame_number, "{}", error_msg);
    handle_error_recovery(error_msg);
  }
}

void MediaFrameAssembler::force_memory_copy_strategy_redetection() {
  if (memory_copy_strategy_detector_) {
    memory_copy_strategy_detector_->reset();
    memory_copy_strategy_detection_active_ = true;
    current_copy_strategy_.reset();
    assembly_controller_->set_strategy(nullptr);
    statistics_.memory_copy_strategy_redetections++;

    ANM_CONFIG_LOG("Memory copy strategy redetection forced");
  }
}

void MediaFrameAssembler::reset() {
  assembly_controller_->reset();

  if (memory_copy_strategy_detector_) {
    memory_copy_strategy_detector_->reset();
    memory_copy_strategy_detection_active_ = config_.enable_memory_copy_strategy_detection &&
                                             !config_.force_contiguous_memory_copy_strategy;
  }

  if (config_.force_contiguous_memory_copy_strategy) {
    current_copy_strategy_ = StrategyFactory::create_contiguous_strategy(
        config_.source_memory_type, config_.destination_memory_type);
    setup_memory_copy_strategy(std::move(current_copy_strategy_));
  } else {
    current_copy_strategy_.reset();
  }

  // Reset statistics (keep cumulative counters)
  statistics_.current_strategy = "UNKNOWN";
  statistics_.current_frame_state = "IDLE";
  statistics_.last_error.clear();

  ANM_CONFIG_LOG("Media Frame assembler has been reset to initial state");
}

MediaFrameAssembler::Statistics MediaFrameAssembler::get_statistics() const {
  // Update current state information
  statistics_.current_frame_state =
      convert_state_to_string(assembly_controller_->get_frame_state());

  if (current_copy_strategy_) {
    statistics_.current_strategy = convert_strategy_to_string(current_copy_strategy_->get_type());
  }

  return statistics_;
}

bool MediaFrameAssembler::has_accumulated_data() const {
  return current_copy_strategy_ && current_copy_strategy_->has_accumulated_data();
}

std::shared_ptr<FrameBufferBase> MediaFrameAssembler::get_current_frame() const {
  return assembly_controller_->get_current_frame();
}

size_t MediaFrameAssembler::get_frame_position() const {
  return assembly_controller_->get_frame_position();
}

StateEvent MediaFrameAssembler::determine_event(const RtpParams& rtp_params, uint8_t* payload) {
  // Check for M-bit marker first
  if (rtp_params.m_bit) {
    if (assembly_controller_->get_frame_state() == FrameState::ERROR_RECOVERY) {
      ANM_STATE_LOG("M-bit detected during error recovery - generating RECOVERY_MARKER event");
      return StateEvent::RECOVERY_MARKER;
    } else {
      return StateEvent::MARKER_DETECTED;
    }
  }

  // Validate packet integrity
  if (!validate_packet_integrity(rtp_params)) {
    return StateEvent::CORRUPTION_DETECTED;
  }

  // Check if we're in memory copy strategy detection phase
  if (memory_copy_strategy_detection_active_ && memory_copy_strategy_detector_) {
    if (memory_copy_strategy_detector_->collect_packet(
            rtp_params, payload, rtp_params.payload_size)) {
      // Enough packets collected, attempt memory copy strategy detection
      auto detected_strategy = memory_copy_strategy_detector_->detect_strategy(
          config_.source_memory_type, config_.destination_memory_type);

      if (detected_strategy) {
        setup_memory_copy_strategy(std::move(detected_strategy));
        memory_copy_strategy_detection_active_ = false;
        return StateEvent::STRATEGY_DETECTED;
      } else {
        // Detection failed, will retry with more packets
        return StateEvent::PACKET_ARRIVED;
      }
    } else {
      // Still collecting packets for detection
      return StateEvent::PACKET_ARRIVED;
    }
  }

  return StateEvent::PACKET_ARRIVED;
}

void MediaFrameAssembler::execute_actions(const StateTransitionResult& result,
                                          const RtpParams& rtp_params, uint8_t* payload) {
  ANM_FRAME_TRACE("execute_actions: should_emit_frame={}, should_complete_frame={}, new_state={}",
                  result.should_emit_frame,
                  result.should_complete_frame,
                  static_cast<int>(result.new_frame_state));
  // Memory copy strategy processing (skip during error recovery as indicated by state machine)
  if (result.new_frame_state == FrameState::RECEIVING_PACKETS &&
      !result.should_skip_memory_copy_processing && current_copy_strategy_ && payload) {
    StateEvent copy_strategy_result = current_copy_strategy_->process_packet(
        *assembly_controller_, payload, rtp_params.payload_size);

    if (copy_strategy_result == StateEvent::CORRUPTION_DETECTED) {
      handle_error_recovery("Memory copy strategy detected corruption");
      return;
    } else if (copy_strategy_result == StateEvent::COPY_EXECUTED) {
      ANM_MEMCOPY_TRACE("Memory copy strategy executed operation successfully");
    }
  }

  // Execute pending copies if requested
  if (result.should_execute_copy && current_copy_strategy_) {
    if (current_copy_strategy_->has_accumulated_data()) {
      StateEvent copy_result =
          current_copy_strategy_->execute_accumulated_copy(*assembly_controller_);
      if (copy_result == StateEvent::CORRUPTION_DETECTED) {
        handle_error_recovery("Copy execution failed");
        return;
      }
    }
  }

  // Handle frame completion
  if (result.should_complete_frame) {
    handle_frame_completion();
  }

  // Handle frame emission
  if (result.should_emit_frame) {
    auto frame = assembly_controller_->get_current_frame();
    if (frame && completion_handler_) {
      ANM_FRAME_TRACE("Emitting frame to completion handler");
      completion_handler_->on_frame_completed(frame);
      // Note: frames_completed is incremented in state controller atomic operation
    }
  }

  // Handle new frame allocation (atomic with frame completion)
  if (result.should_allocate_new_frame) {
    ANM_FRAME_TRACE("Allocating new frame for next packet sequence");
    if (!assembly_controller_->allocate_new_frame()) {
      ANM_FRAME_ERROR(statistics_.current_frame_number,
                      "Failed to allocate new frame after completion");
    } else {
      // Starting a new frame - update statistics
      ANM_STATS_UPDATE(statistics_.current_frame_number++; statistics_.frames_started++;
                       statistics_.packets_in_current_frame = 0;
                       statistics_.bytes_in_current_frame = 0;
                       statistics_.first_sequence_in_frame = 0;);

      ANM_STATS_TRACE("Starting new frame {}", statistics_.current_frame_number);
    }
  }
}

bool MediaFrameAssembler::handle_memory_copy_strategy_detection(const RtpParams& rtp_params,
                                                                uint8_t* payload) {
  if (!memory_copy_strategy_detection_active_ || !memory_copy_strategy_detector_) {
    return true;  // No detection needed or memory copy strategy already available
  }

  // Collect packet for analysis
  if (memory_copy_strategy_detector_->collect_packet(
          rtp_params, payload, rtp_params.payload_size)) {
    // Attempt memory copy strategy detection
    auto detected_strategy = memory_copy_strategy_detector_->detect_strategy(
        config_.source_memory_type, config_.destination_memory_type);

    if (detected_strategy) {
      setup_memory_copy_strategy(std::move(detected_strategy));
      memory_copy_strategy_detection_active_ = false;
      return true;
    } else {
      ANM_STRATEGY_LOG("Strategy detection failed, will retry");
      return false;
    }
  }

  ANM_STRATEGY_LOG("Still collecting packets for strategy detection ({}/{})",
                   memory_copy_strategy_detector_->get_packets_analyzed(),
                   MemoryCopyStrategyDetector::DETECTION_PACKET_COUNT);
  return false;
}

void MediaFrameAssembler::setup_memory_copy_strategy(
    std::unique_ptr<IMemoryCopyStrategy> strategy) {
  current_copy_strategy_ = std::move(strategy);

  // Note: For the old interface compatibility, we would set the memory copy strategy in the
  // assembly controller but since IMemoryCopyStrategy is different from IPacketCopyStrategy, we
  // manage it here

  if (current_copy_strategy_) {
    ANM_CONFIG_LOG(
        "Memory copy strategy setup completed: {}",
        current_copy_strategy_->get_type() == CopyStrategy::CONTIGUOUS ? "CONTIGUOUS" : "STRIDED");
  }
}

bool MediaFrameAssembler::validate_packet_integrity(const RtpParams& rtp_params) {
  auto frame = assembly_controller_->get_current_frame();
  if (!frame) {
    return false;
  }

  int64_t bytes_left = frame->get_size() - assembly_controller_->get_frame_position();

  if (bytes_left < 0) {
    return false;  // Frame overflow
  }

  bool frame_full = (bytes_left == 0);
  if (frame_full && !rtp_params.m_bit) {
    return false;  // Frame full but no marker
  }

  return true;
}

void MediaFrameAssembler::handle_frame_completion() {
  // Execute any accumulated copy operations
  if (current_copy_strategy_ && current_copy_strategy_->has_accumulated_data()) {
    StateEvent copy_result =
        current_copy_strategy_->execute_accumulated_copy(*assembly_controller_);
    if (copy_result == StateEvent::CORRUPTION_DETECTED) {
      handle_error_recovery("Final copy operation failed");
      return;
    }
  }

  // Update frame completion statistics
  ANM_STATS_UPDATE(statistics_.frames_completed++; statistics_.frames_completed_successfully++;
                   statistics_.last_frame_completion_time = std::chrono::steady_clock::now(););

  ANM_STATS_TRACE("Frame {} completed successfully - {} packets, {} bytes",
                  statistics_.current_frame_number,
                  statistics_.packets_in_current_frame,
                  statistics_.bytes_in_current_frame);

  // Reset current frame statistics for next frame
  ANM_STATS_UPDATE(statistics_.packets_in_current_frame = 0; statistics_.bytes_in_current_frame = 0;
                   statistics_.first_sequence_in_frame = 0;);
}

void MediaFrameAssembler::handle_error_recovery(const std::string& error_message) {
  ANM_STATS_UPDATE(statistics_.last_error = error_message; statistics_.errors_recovered++;
                   statistics_.error_recovery_cycles++;
                   statistics_.frames_dropped++;
                   statistics_.last_error_time = std::chrono::steady_clock::now(););

  // Log dropped frame information
  ANM_FRAME_WARN(
      statistics_.current_frame_number,
      "Error recovery initiated: {} - discarding {} packets, {} bytes - waiting for M-bit marker",
      error_message,
      statistics_.packets_in_current_frame,
      statistics_.bytes_in_current_frame);

  if (completion_handler_) {
    completion_handler_->on_frame_error(error_message);
  }

  // Reset memory copy strategy if needed
  if (current_copy_strategy_) {
    current_copy_strategy_->reset();
  }

  // Reset current frame statistics since frame is being dropped
  ANM_STATS_UPDATE(statistics_.packets_in_current_frame = 0; statistics_.bytes_in_current_frame = 0;
                   statistics_.first_sequence_in_frame = 0;);
}

void MediaFrameAssembler::update_statistics(StateEvent event) {
  switch (event) {
    case StateEvent::PACKET_ARRIVED:
      statistics_.packets_processed++;
      ANM_STATS_UPDATE(statistics_.packets_in_current_frame++);
      break;
    case StateEvent::STRATEGY_DETECTED:
      ANM_STATS_UPDATE(statistics_.memory_copy_strategy_redetections++);
      break;
    case StateEvent::MARKER_DETECTED:
      // Frame completion handled elsewhere
      break;
    case StateEvent::RECOVERY_MARKER:
      // Recovery completion handled elsewhere
      break;
    case StateEvent::CORRUPTION_DETECTED:
      ANM_STATS_UPDATE(statistics_.memory_corruption_errors++);
      break;
    default:
      break;
  }
}

void MediaFrameAssembler::update_packet_statistics(const RtpParams& rtp_params) {
  ANM_STATS_UPDATE(
      // Check for sequence discontinuity (only if we have a previous sequence number)
      if (statistics_.last_sequence_number != 0 && statistics_.packets_processed > 1) {
        uint32_t expected_seq = statistics_.last_sequence_number + 1;
        if (rtp_params.sequence_number != expected_seq) {
          statistics_.sequence_discontinuities++;

          // Check for potential buffer overflow (large gaps)
          int32_t gap =
              static_cast<int32_t>(rtp_params.sequence_number) - static_cast<int32_t>(expected_seq);

          // Power-of-2 check for buffer wraparound (524288 = 2^19)
          if (gap > kRtpSequenceGapWraparoundThreshold && (gap & (gap - 1)) == 0) {
            statistics_.buffer_overflow_errors++;
            ANM_FRAME_TRACE(
                "Frame {}: Potential RX buffer wraparound detected: RTP sequence gap {} "
                "(2^{}) - processing pipeline too slow, cannot keep up with incoming data rate",
                statistics_.current_frame_number,
                gap,
                __builtin_ctz(gap));
          } else {
            ANM_FRAME_TRACE(
                "Frame {}: RTP sequence discontinuity detected: expected {}, got {} (gap of {})",
                statistics_.current_frame_number,
                expected_seq,
                rtp_params.sequence_number,
                gap);
          }
        }
      }

      // Update sequence tracking
      statistics_.last_sequence_number = rtp_params.sequence_number;

      // Set first sequence in frame if this is the first packet
      if (statistics_.first_sequence_in_frame == 0) {
        statistics_.first_sequence_in_frame = rtp_params.sequence_number;
      }

      // Update byte count
      statistics_.bytes_in_current_frame += rtp_params.payload_size;);
}

// ========================================================================================
// MediaFrameAssembler Statistics Implementation
// ========================================================================================

std::string MediaFrameAssembler::get_statistics_summary() const {
#if ENABLE_STATISTICS_LOGGING
  std::ostringstream ss;
  auto stats = get_statistics();

  ss << "MediaFrameAssembler Statistics Summary:\n";
  ss << "========================================\n";

  // Basic counters
  ss << "Basic Counters:\n";
  ss << "  Packets processed: " << stats.packets_processed << "\n";
  ss << "  Frames completed: " << stats.frames_completed << "\n";
  ss << "  Errors recovered: " << stats.errors_recovered << "\n";
  ss << "  Strategy redetections: " << stats.memory_copy_strategy_redetections << "\n";

  // Enhanced frame tracking
  ss << "\nFrame Tracking:\n";
  ss << "  Current frame number: " << stats.current_frame_number << "\n";
  ss << "  Frames started: " << stats.frames_started << "\n";
  ss << "  Frames dropped: " << stats.frames_dropped << "\n";
  ss << "  Frames completed successfully: " << stats.frames_completed_successfully << "\n";
  if (stats.frames_started > 0) {
    double completion_rate =
        (double)stats.frames_completed_successfully / stats.frames_started * 100.0;
    ss << "  Frame completion rate: " << std::fixed << std::setprecision(2) << completion_rate
       << "%\n";
  }

  // Enhanced error tracking
  ss << "\nError Tracking:\n";
  ss << "  Sequence discontinuities: " << stats.sequence_discontinuities << "\n";
  ss << "  Buffer overflow errors: " << stats.buffer_overflow_errors << "\n";
  ss << "  Memory corruption errors: " << stats.memory_corruption_errors << "\n";
  ss << "  Error recovery cycles: " << stats.error_recovery_cycles << "\n";

  // Current frame metrics
  ss << "\nCurrent Frame:\n";
  ss << "  Packets in current frame: " << stats.packets_in_current_frame << "\n";
  ss << "  Bytes in current frame: " << stats.bytes_in_current_frame << "\n";
  ss << "  Last sequence number: " << stats.last_sequence_number << "\n";
  ss << "  First sequence in frame: " << stats.first_sequence_in_frame << "\n";

  // State information
  ss << "\nState Information:\n";
  ss << "  Current strategy: " << stats.current_strategy << "\n";
  ss << "  Current frame state: " << stats.current_frame_state << "\n";
  if (!stats.last_error.empty()) {
    ss << "  Last error: " << stats.last_error << "\n";
  }

  // Timing information
  auto now = std::chrono::steady_clock::now();
  if (stats.last_frame_completion_time != std::chrono::steady_clock::time_point{}) {
    auto time_since_last_frame = std::chrono::duration_cast<std::chrono::milliseconds>(
                                     now - stats.last_frame_completion_time)
                                     .count();
    ss << "\nTiming:\n";
    ss << "  Time since last frame completion: " << time_since_last_frame << " ms\n";
  }

  if (stats.last_error_time != std::chrono::steady_clock::time_point{}) {
    auto time_since_last_error =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - stats.last_error_time).count();
    ss << "  Time since last error: " << time_since_last_error << " ms\n";
  }

  return ss.str();
#else
  return "Enhanced statistics disabled for performance (compile with ENABLE_STATISTICS_LOGGING)";
#endif
}

// ========================================================================================
// DefaultFrameCompletionHandler Implementation
// ========================================================================================

DefaultFrameCompletionHandler::DefaultFrameCompletionHandler(
    std::function<void(std::shared_ptr<FrameBufferBase>)> frame_ready_callback,
    std::function<void(const std::string&)> error_callback)
    : frame_ready_callback_(frame_ready_callback), error_callback_(error_callback) {}

void DefaultFrameCompletionHandler::on_frame_completed(std::shared_ptr<FrameBufferBase> frame) {
  if (frame_ready_callback_) {
    frame_ready_callback_(frame);
  }
}

void DefaultFrameCompletionHandler::on_frame_error(const std::string& error_message) {
  if (error_callback_) {
    error_callback_(error_message);
  } else {
    ANM_LOG_ERROR("Frame processing error: {}", error_message);
  }
}

// ========================================================================================
// AssemblerConfigurationHelper Implementation
// ========================================================================================

AssemblerConfiguration AssemblerConfigurationHelper::create_with_burst_parameters(
    size_t header_stride, size_t payload_stride, bool hds_enabled, bool payload_on_cpu,
    bool frames_on_host) {
  AssemblerConfiguration config;

  config.header_stride_size = header_stride;
  config.payload_stride_size = payload_stride;
  config.hds_enabled = hds_enabled;

  config.source_memory_type = payload_on_cpu ? nvidia::gxf::MemoryStorageType::kHost
                                             : nvidia::gxf::MemoryStorageType::kDevice;

  config.destination_memory_type = frames_on_host ? nvidia::gxf::MemoryStorageType::kHost
                                                  : nvidia::gxf::MemoryStorageType::kDevice;

  config.enable_memory_copy_strategy_detection = true;
  config.force_contiguous_memory_copy_strategy = false;

  return config;
}

AssemblerConfiguration AssemblerConfigurationHelper::create_test_config(bool force_contiguous) {
  AssemblerConfiguration config;

  config.source_memory_type = nvidia::gxf::MemoryStorageType::kHost;
  config.destination_memory_type = nvidia::gxf::MemoryStorageType::kHost;
  config.hds_enabled = false;
  config.header_stride_size = 1500;
  config.payload_stride_size = 1500;
  config.force_contiguous_memory_copy_strategy = force_contiguous;
  config.enable_memory_copy_strategy_detection = !force_contiguous;

  return config;
}

bool AssemblerConfigurationHelper::validate_configuration(const AssemblerConfiguration& config) {
  // Basic validation
  if (config.enable_memory_copy_strategy_detection &&
      config.force_contiguous_memory_copy_strategy) {
    ANM_LOG_ERROR(
        "Configuration error: Cannot enable memory copy strategy detection and force contiguous "
        "strategy "
        "simultaneously");
    return false;
  }

  // Stride validation (if detection is enabled)
  if (config.enable_memory_copy_strategy_detection) {
    if (config.header_stride_size == 0 && config.payload_stride_size == 0) {
      ANM_LOG_WARN(
          "Zero stride sizes with strategy detection enabled may affect detection accuracy");
    }
  }

  return true;
}

}  // namespace holoscan::ops
