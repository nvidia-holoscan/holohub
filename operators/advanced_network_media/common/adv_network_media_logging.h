/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Advanced Network Media Operator - Unified Logging Control System
 *
 * This header provides centralized control over all logging in the advanced_network_media operator.
 * Different logging categories can be independently enabled/disabled for performance optimization.
 */

#ifndef OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_ADV_NETWORK_MEDIA_LOGGING_H_
#define OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_ADV_NETWORK_MEDIA_LOGGING_H_

#include <holoscan/logger/logger.hpp>

// ========================================================================================
// LOGGING CONTROL CONFIGURATION
// ========================================================================================
//
// The advanced network media operator supports multiple logging levels for different
// use cases and performance requirements:
//
// 1. CRITICAL LOGS (Always Enabled):
//    - Errors that indicate operator malfunction
//    - Warnings about configuration issues or data problems
//    - Essential initialization/configuration messages
//
// 2. PACKET/POINTER TRACING (ENABLE_PACKET_TRACING):
//    - Per-packet processing details
//    - Memory pointer tracking and validation
//    - Low-level RTP packet analysis
//    - Buffer management operations
//    - PERFORMANCE IMPACT: High (5-15% overhead)
//
// 3. STATISTICS AND MONITORING (ENABLE_STATISTICS_LOGGING):
//    - Frame completion statistics
//    - Throughput and performance metrics
//    - Periodic statistics reports
//    - Enhanced error tracking and analytics
//    - PERFORMANCE IMPACT: Medium (1-5% overhead)
//
// 4. CONFIGURATION AND STATE (ENABLE_CONFIG_LOGGING):
//    - Strategy detection results
//    - State machine transitions
//    - Memory copy strategy configuration
//    - Burst parameter updates
//    - PERFORMANCE IMPACT: Low (0.1-1% overhead)
//
// PRODUCTION DEPLOYMENT RECOMMENDATIONS:
// - Comment out all ENABLE_* flags for maximum performance
// - Keep only critical error/warning logs
// - Use for high-throughput production environments
//
// DEVELOPMENT/DEBUGGING RECOMMENDATIONS:
// - Frame assembly issues: Enable ENABLE_FRAME_TRACING + ENABLE_MEMCOPY_TRACING
// - Network/frame processing: Enable ENABLE_FRAME_TRACING + ENABLE_PACKET_TRACING
// - Strategy detection problems: Enable ENABLE_STRATEGY_TRACING + ENABLE_MEMCOPY_TRACING
// - State machine issues: Enable ENABLE_STATE_LOGGING + ENABLE_FRAME_TRACING
// - Deep state debugging: Enable ENABLE_STATE_TRACING + ENABLE_STATE_LOGGING
// - Configuration issues: Enable ENABLE_CONFIG_LOGGING
// - Performance analysis: Enable ENABLE_STATISTICS_LOGGING + ENABLE_FRAME_TRACING
//
// TROUBLESHOOTING SPECIFIC ISSUES:
// - Packet loss/corruption: ENABLE_PACKET_TRACING + ENABLE_FRAME_TRACING
// - Frame drop/corruption: ENABLE_FRAME_TRACING + ENABLE_MEMCOPY_TRACING
// - Memory copy errors: ENABLE_MEMCOPY_TRACING only
// - Strategy detection failures: ENABLE_STRATEGY_TRACING only
// - Strategy vs execution: ENABLE_STRATEGY_TRACING + ENABLE_MEMCOPY_TRACING
// - State transitions/recovery: ENABLE_STATE_LOGGING only
// - State validation/internals: ENABLE_STATE_TRACING only
// - Complex state issues: ENABLE_STATE_LOGGING + ENABLE_STATE_TRACING
// - Setup/parameter issues: ENABLE_CONFIG_LOGGING only
// - Performance bottlenecks: ENABLE_STATISTICS_LOGGING + ENABLE_FRAME_TRACING
//

// ========================================================================================
// LOGGING LEVEL CONTROLS (0 = disabled, 1 = enabled)
// ========================================================================================

// Packet-level tracing (highest performance impact)
// Includes: individual packet processing, pointer tracking, buffer operations
#define ENABLE_PACKET_TRACING 0

// Frame-level tracing (medium-high performance impact)
// Includes: frame allocation, emission, completion, lifecycle events
#define ENABLE_FRAME_TRACING 0

// Memory copy tracing (medium performance impact)
// Includes: packet-to-frame assembly, memory copy strategy operations
#define ENABLE_MEMCOPY_TRACING 0

// Strategy detection tracing (low-medium performance impact)
// Includes: memory copy strategy detection, pattern analysis, strategy switching
#define ENABLE_STRATEGY_TRACING 1

// State machine logging/tracing (medium performance impact)
// Includes: state transitions, state validation, internal state debugging, error recovery
#define ENABLE_STATE_LOGGING 0

// Statistics and monitoring (medium performance impact)
// Includes: frame completion stats, throughput metrics, periodic reports
#define ENABLE_STATISTICS_LOGGING 1

// Verbose statistics logging (high performance impact)
// Includes: per-frame detailed logging (very verbose - use only for debugging)
#define ENABLE_VERBOSE_STATISTICS 0

// Performance logging (medium-high performance impact)
// Includes: timing measurements, performance profiling, compute duration tracking
#define ENABLE_PERFORMANCE_LOGGING 0

// Configuration logging (low performance impact)
// Includes: parameter updates, operator setup, initialization
#define ENABLE_CONFIG_LOGGING 1

// ========================================================================================
// LOGGING MACRO DEFINITIONS
// ========================================================================================

// Critical logs - ALWAYS ENABLED (errors, warnings, essential info)
#define ANM_LOG_ERROR(fmt, ...) HOLOSCAN_LOG_ERROR("[ANM] " fmt, ##__VA_ARGS__)
#define ANM_LOG_WARN(fmt, ...) HOLOSCAN_LOG_WARN("[ANM] " fmt, ##__VA_ARGS__)
#define ANM_LOG_CRITICAL(fmt, ...) HOLOSCAN_LOG_CRITICAL("[ANM] " fmt, ##__VA_ARGS__)
#define ANM_LOG_INFO(fmt, ...) HOLOSCAN_LOG_INFO("[ANM] " fmt, ##__VA_ARGS__)

// Packet/Pointer tracing - CONDITIONAL (per-packet details, pointer tracking)
#if ENABLE_PACKET_TRACING
#define ANM_PACKET_TRACE(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_PACKET] " fmt, ##__VA_ARGS__)
#define ANM_POINTER_TRACE(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_PTR] " fmt, ##__VA_ARGS__)
#else
#define ANM_PACKET_TRACE(fmt, ...) \
  do {                             \
  } while (0)
#define ANM_POINTER_TRACE(fmt, ...) \
  do {                              \
  } while (0)
#endif

// Statistics and monitoring - CONDITIONAL (frame stats, metrics)
#if ENABLE_STATISTICS_LOGGING
#define ANM_STATS_LOG(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_STATS] " fmt, ##__VA_ARGS__)
#define ANM_STATS_UPDATE(code) \
  do { code; } while (0)
#else
#define ANM_STATS_LOG(fmt, ...) \
  do {                          \
  } while (0)
#define ANM_STATS_UPDATE(code) \
  do {                         \
  } while (0)
#endif

// Statistics tracing - CONDITIONAL (per-frame detailed logging)
#if ENABLE_VERBOSE_STATISTICS
#define ANM_STATS_TRACE(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_STATS] " fmt, ##__VA_ARGS__)
#else
#define ANM_STATS_TRACE(fmt, ...) \
  do {                            \
  } while (0)
#endif

// Performance logging - CONDITIONAL (timing measurements, profiling)
#if ENABLE_PERFORMANCE_LOGGING
#define ANM_PERF_LOG(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_PERF] " fmt, ##__VA_ARGS__)
#else
#define ANM_PERF_LOG(fmt, ...) \
  do {                         \
  } while (0)
#endif

// Configuration logging - CONDITIONAL (operator setup, parameters)
#if ENABLE_CONFIG_LOGGING
#define ANM_CONFIG_LOG(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_CONFIG] " fmt, ##__VA_ARGS__)
#else
#define ANM_CONFIG_LOG(fmt, ...) \
  do {                           \
  } while (0)
#endif

// State machine logging/tracing - CONDITIONAL (state transitions, validation, debugging)
#if ENABLE_STATE_LOGGING
#define ANM_STATE_LOG(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_STATE] " fmt, ##__VA_ARGS__)
#define ANM_STATE_TRACE(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_STATE] " fmt, ##__VA_ARGS__)
#else
#define ANM_STATE_LOG(fmt, ...) \
  do {                          \
  } while (0)
#define ANM_STATE_TRACE(fmt, ...) \
  do {                            \
  } while (0)
#endif

// ========================================================================================
// SPECIALIZED LOGGING HELPERS
// ========================================================================================

// Error with frame context (always enabled for critical errors)
#define ANM_FRAME_ERROR(frame_num, fmt, ...) \
  ANM_LOG_ERROR("Frame {} - " fmt, frame_num, ##__VA_ARGS__)

// Warning with frame context (always enabled)
#define ANM_FRAME_WARN(frame_num, fmt, ...) \
  ANM_LOG_WARN("Frame {} - " fmt, frame_num, ##__VA_ARGS__)

// Frame-level tracing (independent control)
#if ENABLE_FRAME_TRACING
#define ANM_FRAME_TRACE(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_FRAME] " fmt, ##__VA_ARGS__)
#else
#define ANM_FRAME_TRACE(fmt, ...) \
  do {                            \
  } while (0)
#endif

// Memory copy operation tracing (independent control)
#if ENABLE_MEMCOPY_TRACING
#define ANM_MEMCOPY_TRACE(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_MEMCOPY] " fmt, ##__VA_ARGS__)
#else
#define ANM_MEMCOPY_TRACE(fmt, ...) \
  do {                              \
  } while (0)
#endif

// Strategy detection tracing (independent control)
#if ENABLE_STRATEGY_TRACING
#define ANM_STRATEGY_LOG(fmt, ...) HOLOSCAN_LOG_INFO("[ANM_STRATEGY] " fmt, ##__VA_ARGS__)
#else
#define ANM_STRATEGY_LOG(fmt, ...) \
  do {                             \
  } while (0)
#endif

#endif  // OPERATORS_ADVANCED_NETWORK_MEDIA_COMMON_ADV_NETWORK_MEDIA_LOGGING_H_
