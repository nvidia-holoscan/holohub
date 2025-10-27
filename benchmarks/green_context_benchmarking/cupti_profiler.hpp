/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUPTI_PROFILER_HPP
#define CUPTI_PROFILER_HPP

#include <cstring>
#include <cupti.h>
#include <iostream>
#include <mutex>
#include <unordered_map>

// CUPTI Scheduling Latency Measurement Infrastructure
namespace cupti_timing {

// Structure to hold kernel launch timing data
struct KernelLaunchData {
  uint64_t launch_timestamp;
};

// Global state for CUPTI measurements
class CuptiSchedulingProfiler {
 private:
  static CuptiSchedulingProfiler* instance_;
  static std::mutex mutex_;

  CUpti_SubscriberHandle subscriber_;
  std::unordered_map<uint32_t, KernelLaunchData> launch_map_;  // correlationId -> launch data
  std::unordered_map<uint32_t, double> scheduling_latencies_;  // correlationId -> latency (us)
  std::unordered_map<uint32_t, double> execution_durations_;  // correlationId -> execution duration
  std::mutex data_mutex_;
  bool initialized_;

 public:
  static CuptiSchedulingProfiler* getInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ == nullptr) {
      instance_ = new CuptiSchedulingProfiler();
    }
    return instance_;
  }

  CuptiSchedulingProfiler() : initialized_(false) {}

  ~CuptiSchedulingProfiler() {
    if (initialized_) {
      cleanup();
    }
  }

  bool initialize() {
    if (initialized_)
      return true;

    try {
      // Subscribe to CUPTI callbacks
      CUptiResult result = cuptiSubscribe(&subscriber_, (CUpti_CallbackFunc)apiCallback, this);
      if (result != CUPTI_SUCCESS) {
        std::cerr << "[CUPTI] Failed to subscribe to callbacks" << std::endl;
        return false;
      }

      // Enable callback domain for CUDA runtime API
      result = cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API);
      if (result != CUPTI_SUCCESS) {
        std::cerr << "[CUPTI] Failed to enable runtime API domain" << std::endl;
        cuptiUnsubscribe(subscriber_);
        return false;
      }

      // Enable activity for kernel execution tracing
      result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
      if (result != CUPTI_SUCCESS) {
        std::cerr << "[CUPTI] Failed to enable concurrent kernel activity" << std::endl;
        cuptiUnsubscribe(subscriber_);
        return false;
      }

      // Register activity flush callback
      result = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
      if (result != CUPTI_SUCCESS) {
        std::cerr << "[CUPTI] Failed to register activity callbacks" << std::endl;
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        cuptiUnsubscribe(subscriber_);
        return false;
      }

      initialized_ = true;
      std::cout << "[CUPTI] Successfully initialized scheduling latency profiler" << std::endl;
      return true;
    } catch (const std::exception& e) {
      std::cerr << "[CUPTI] Initialization error: " << e.what() << std::endl;
      return false;
    }
  }

  void cleanup() {
    if (!initialized_)
      return;

    // Flush remaining activity records
    cuptiActivityFlushAll(0);

    // Disable activities and callbacks
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiUnsubscribe(subscriber_);
    initialized_ = false;
  }

  // Check if measurements are available (thread-safe)
  bool hasMeasurements() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return !scheduling_latencies_.empty();
  }

  // Check if there are pending launches waiting for activity records (thread-safe)
  bool hasPendingLaunches() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return !launch_map_.empty();
  }

  // Get latest scheduling latency measurement and clear others
  double getLatestSchedulingLatency() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (scheduling_latencies_.empty()) {
      if (!launch_map_.empty()) {
        std::cout << "[CUPTI] WARNING: " << launch_map_.size()
                  << " kernel launches detected but no GPU activity records matched!"
                  << std::endl;
      }
      return -1.0;
    }

    // Find the measurement with the highest correlation ID (most recent)
    auto max_it = std::max_element(
        scheduling_latencies_.begin(),
        scheduling_latencies_.end(),
        [](const std::pair<uint32_t, double>& a, const std::pair<uint32_t, double>& b) {
          return a.first < b.first;  // Compare correlation IDs
        });

    double latest_latency = max_it->second;

    // Clear all measurements to prevent accumulation
    scheduling_latencies_.clear();
    launch_map_.clear();

    return latest_latency;
  }

  // Get latest kernel execution duration
  double getLatestExecutionDuration() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (execution_durations_.empty()) {
      return -1.0;  // No execution duration available
    }

    // Find the measurement with the highest correlation ID (most recent)
    auto max_it = std::max_element(
        execution_durations_.begin(),
        execution_durations_.end(),
        [](const std::pair<uint32_t, double>& a, const std::pair<uint32_t, double>& b) {
          return a.first < b.first;  // Compare correlation IDs
        });

    double latest_duration = max_it->second;

    // Clear execution durations
    execution_durations_.clear();

    return latest_duration;
  }

 private:
  // CUPTI API callback for kernel launches
  static void CUPTIAPI apiCallback(void* userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId callbackId, const CUpti_CallbackData* cbdata) {
    CuptiSchedulingProfiler* profiler = (CuptiSchedulingProfiler*)userdata;

    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
        (callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
          callbackId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020)) {
      if (cbdata->callbackSite == CUPTI_API_ENTER) {
        // Record kernel launch timestamp
        uint64_t timestamp;
        cuptiGetTimestamp(&timestamp);

        // Store launch data - kernel filtering happens later in activity processing
        std::lock_guard<std::mutex> lock(profiler->data_mutex_);
        KernelLaunchData data;
        data.launch_timestamp = timestamp;

        profiler->launch_map_[cbdata->correlationId] = data;
      }
    }
  }

  // CUPTI activity buffer management - increased buffer size for high contention
  static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 64 * 1024;  // Increased to 64KB buffer for better handling under load
    *buffer = (uint8_t*)malloc(*size);
    *maxNumRecords = 0;

    if (*buffer == nullptr) {
      std::cerr << "[CUPTI] ERROR: Failed to allocate activity buffer!" << std::endl;
      *size = 0;
    }
  }

  static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                                      size_t size, size_t validSize) {
    CuptiSchedulingProfiler* profiler = CuptiSchedulingProfiler::getInstance();

    if (validSize > 0) {
      // Check for potential buffer overflow
      if (validSize == size) {
        std::cout
            << "[CUPTI] WARNING: Activity buffer may have overflowed (validSize == bufferSize: "
            << validSize << ")" << std::endl;
      }
      profiler->processActivityBuffer(buffer, validSize);
    } else {
      std::cout << "[CUPTI] WARNING: Empty activity buffer received" << std::endl;
    }
    free(buffer);
  }

  void processActivityBuffer(uint8_t* buffer, size_t validSize) {
    CUpti_Activity* record = nullptr;
    int timing_kernels_found = 0;
    int matches_found = 0;

    std::lock_guard<std::mutex> lock(data_mutex_);

    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        CUpti_ActivityKernel9* kernelRecord = (CUpti_ActivityKernel9*)record;

        // Filter by kernel name - only process timing benchmark kernels
        bool is_timing_kernel = false;
        if (kernelRecord->name) {
          const char* kernel_name = kernelRecord->name;
          is_timing_kernel = (strstr(kernel_name, "simple_benchmark_kernel") != nullptr);
        }

        if (is_timing_kernel) {
          timing_kernels_found++;

          auto it = launch_map_.find(kernelRecord->correlationId);
          if (it != launch_map_.end()) {
            matches_found++;

            // Calculate scheduling latency: GPU execution start - CPU launch time
            double latency_ns = (double)(kernelRecord->start - it->second.launch_timestamp);
            double latency_us = latency_ns / 1000.0;  // Convert to microseconds

            // Calculate kernel execution duration: GPU execution end - GPU execution start
            double execution_duration_ns = (double)(kernelRecord->end - kernelRecord->start);
            double execution_duration_us = execution_duration_ns / 1000.0;

            // Sanity check - latency should be reasonable (0.1μs to 10ms)
            if (latency_us >= 0.1 && latency_us <= 10000.0) {
              scheduling_latencies_[kernelRecord->correlationId] = latency_us;
              execution_durations_[kernelRecord->correlationId] = execution_duration_us;
            } else {
              std::cout << "[CUPTI] WARNING: Rejected latency " << latency_us
                        << " μs (expected 0.1-10000 μs)" << std::endl;
            }

            // Remove from launch_map to save memory
            launch_map_.erase(it);
          }
        } else {
          // This was a background kernel, remove from launch_map if present
          auto it = launch_map_.find(kernelRecord->correlationId);
          if (it != launch_map_.end()) {
            launch_map_.erase(it);  // Clean up background kernel launch data
          }
        }
      }
    }

    // Optional: Log only if there are issues
    if (matches_found == 0 && timing_kernels_found > 0) {
      std::cout << "[CUPTI] WARNING: Found " << timing_kernels_found
                << " timing kernels but no matches in activity buffer" << std::endl;
    }
  }
};

// Static member definitions
CuptiSchedulingProfiler* CuptiSchedulingProfiler::instance_ = nullptr;
std::mutex CuptiSchedulingProfiler::mutex_;

}  // namespace cupti_timing

#endif
