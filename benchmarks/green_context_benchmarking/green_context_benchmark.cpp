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

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// CUPTI headers
#include <cupti.h>
#include "cupti_profiler.hpp"
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_macros.hpp>
#include "benchmark_cuda_kernel.cu.hpp"

using namespace holoscan;

struct BenchmarkStats {
  double avg = 0.0;
  double std_dev = 0.0;
  double min_val = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max_val = 0.0;
  size_t sample_count = 0;
  std::vector<double> sorted_data;
};

// Calculate percentiles from sorted data
double calculate_percentile(const std::vector<double>& sorted_data, double percentile) {
  if (sorted_data.empty())
    return 0.0;

  double index = (percentile / 100.0) * (sorted_data.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(index));
  size_t upper = static_cast<size_t>(std::ceil(index));

  if (lower == upper) {
    return sorted_data[lower];
  }

  double weight = index - lower;
  return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
}

// Calculate standard deviation
double calculate_std_dev(const std::vector<double>& data, double mean) {
  if (data.size() <= 1)
    return 0.0;

  double sum_sq_diff = 0.0;
  for (double value : data) {
    double diff = value - mean;
    sum_sq_diff += diff * diff;
  }

  return std::sqrt(sum_sq_diff / (data.size() - 1));
}

// Calculate benchmark statistics
BenchmarkStats calculate_benchmark_stats(
  const std::vector<double>& raw_values, bool skip_negative_values = false) {
  BenchmarkStats stats;

  // Extract values
  for (const auto& value : raw_values) {
    if (value >= 0.0 || !skip_negative_values) {
      stats.sorted_data.push_back(value);
    }
  }

  if (stats.sorted_data.empty())
    return stats;

  std::sort(stats.sorted_data.begin(), stats.sorted_data.end());
  stats.sample_count = stats.sorted_data.size();

  // Calculate basic statistics
  stats.avg =
      std::accumulate(stats.sorted_data.begin(), stats.sorted_data.end(), 0.0) / stats.sample_count;
  stats.std_dev = calculate_std_dev(stats.sorted_data, stats.avg);

  // Calculate percentiles
  stats.min_val = stats.sorted_data.front();
  stats.max_val = stats.sorted_data.back();
  stats.p50 = calculate_percentile(stats.sorted_data, 50.0);
  stats.p95 = calculate_percentile(stats.sorted_data, 95.0);
  stats.p99 = calculate_percentile(stats.sorted_data, 99.0);

  return stats;
}

// Calculate empirical CDF at a given value
double calculate_cdf(const std::vector<double>& data, double x) {
  if (data.empty())
    return 0.0;

  int count = 0;
  for (double value : data) {
    if (value <= x)
      count++;
  }

  return static_cast<double>(count) / data.size();
}

class DummyLoadOp : public Operator {
 private:
  Parameter<int> load_intensity_;
  Parameter<int> workload_size_;
  Parameter<int> threads_per_block_;
  float* d_load_data_;
  cudaStream_t cached_stream_ = nullptr;
  std::vector<double> execution_times_us_;  // Store execution times for statistics
  mutable std::mutex timing_mutex_;      // Protect timing data access (mutable for const methods)

 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyLoadOp)

  DummyLoadOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(
        load_intensity_, "load_intensity", "Load Intensity", "GPU load intensity factor", 100);
    spec.param(workload_size_, "workload_size", "Workload Size", "Size of GPU workload", 262144);
    spec.param(threads_per_block_,
               "threads_per_block",
               "Threads Per Block",
               "CUDA threads per block for GPU kernel",
               512);
  }

  void initialize() override {
    Operator::initialize();

    int load_intensity = load_intensity_.get();
    int workload_size = workload_size_.get();

    // Initialize CUDA resources
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&d_load_data_, workload_size * sizeof(float)),
                                   "cudaMalloc failed");

    // Initialize with random data
    std::vector<float> host_data(workload_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (auto& val : host_data) {
      val = dis(gen);
    }

    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(
            d_load_data_, host_data.data(), workload_size * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy failed");

    HOLOSCAN_LOG_INFO("[DummyLoadOp] GPU load initialized: {} elements, intensity {}",
      workload_size, load_intensity);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto tick_start_time = std::chrono::steady_clock::now();

    if (cached_stream_ == nullptr) {
      auto maybe_stream = context.allocate_cuda_stream("dummy_load_stream");
      if (!maybe_stream) {
        HOLOSCAN_LOG_ERROR("[DummyLoadOp] Failed to allocate non-default CUDA stream");
        throw std::runtime_error("Failed to allocate non-default CUDA stream");
      }
      cached_stream_ = maybe_stream.value();
      HOLOSCAN_LOG_INFO("[DummyLoadOp] Using allocated non-default CUDA stream: {}",
        reinterpret_cast<long long>(cached_stream_));
    }

    run_background_load_kernel();

    auto tick_end_time = std::chrono::steady_clock::now();
    auto tick_duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(tick_end_time - tick_start_time)
            .count();

    {
      std::lock_guard<std::mutex> lock(timing_mutex_);
      execution_times_us_.push_back(static_cast<double>(tick_duration_us));
    }
  }

  BenchmarkStats get_compute_execution_time_benchmark_stats() const {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    BenchmarkStats benchmark_stats;

    if (execution_times_us_.empty()) {
      return benchmark_stats;
    }
    benchmark_stats = calculate_benchmark_stats(execution_times_us_, false);

    return benchmark_stats;
  }

  ~DummyLoadOp() {
    if (d_load_data_) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(d_load_data_), "cudaFree failed");
    }
  }

 private:
  void run_background_load_kernel() {
    int workload_size = workload_size_.get();
    int load_intensity = load_intensity_.get();

    // Launch background load kernel on non-default stream
    int threads_per_block = threads_per_block_.get();
    int blocks = (workload_size + threads_per_block - 1) / threads_per_block;

    // Launch background load to create GPU contention
    async_run_background_load_kernel(d_load_data_, workload_size, load_intensity,
      threads_per_block, cached_stream_);
    cudaStreamSynchronize(cached_stream_);
  }
};

class TimingBenchmarkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimingBenchmarkOp)

  TimingBenchmarkOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(
        workload_size_, "workload_size", "Workload Size", "Size of timing workload", 1024);
    spec.param(total_samples_,
               "total_samples",
               "Total Samples",
               "Total samples to measure in the benchmark",
               100);
  }

  void initialize() override {
    Operator::initialize();
    execution_count_ = 0;

    // Initialize CUDA kernel parameters
    threads_per_block_ = 256;
    blocks_ = (workload_size_.get() + threads_per_block_ - 1) / threads_per_block_;

    // Allocate CUDA memory for CUDA kernel
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMalloc(&d_benchmark_data_, workload_size_.get() * sizeof(float)), "cudaMalloc failed");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemset(d_benchmark_data_, 0, workload_size_.get() * sizeof(float)), "cudaMemset failed");

    // Initialize CUPTI profiler for CUDA kernel launch-start time measurement
    cupti_profiler_ = cupti_timing::CuptiSchedulingProfiler::getInstance();
    if (!cupti_profiler_->initialize()) {
      HOLOSCAN_LOG_WARN("[TimingBenchmarkOp] CUPTI initialization failed, no "
                        "CUDA kernel launch-start time measurements will be available");
      cupti_profiler_ = nullptr;
    }
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    execution_count_++;

    // Allocate non-default CUDA stream for this operator when the first time the
    // operator is executed
    if (cached_stream_ == nullptr) {
      auto maybe_stream = context.allocate_cuda_stream("timing_stream");
      if (!maybe_stream) {
        HOLOSCAN_LOG_ERROR("[TimingBenchmarkOp] Failed to allocate non-default CUDA stream");
        throw std::runtime_error("Failed to allocate non-default CUDA stream");
      }
      cached_stream_ = maybe_stream.value();
      HOLOSCAN_LOG_INFO("[TimingBenchmarkOp] Using allocated non-default CUDA stream: {}",
        reinterpret_cast<long long>(cached_stream_));
    }

    // Print out log every 1/10 of total samples to not spam the console
    int log_output_interval = std::max(1, total_samples_.get() / 10);
    if (execution_count_ % log_output_interval == 0 || execution_count_ == 1) {
      HOLOSCAN_LOG_INFO("[TimingBenchmarkOp] Collecting {}/{} samples",
                        execution_count_, total_samples_.get());
    }

    // Launch kernel - CUPTI will capture launch and execution timestamps
    async_run_simple_benchmark_kernel(d_benchmark_data_, workload_size_.get(),
                                       threads_per_block_, cached_stream_);
    cudaStreamSynchronize(cached_stream_);

    double cuda_kernel_launch_start_time = -1.0;
    double cuda_kernel_execution_time = -1.0;
    if (cupti_profiler_) {
      // Force flush CUPTI activities to trigger processing
      cuptiActivityFlushAll(0);

      // Enhanced polling with adaptive backoff for high contention scenarios
      const int max_poll_attempts = 500;       // Increased from 100 for high contention
      const int initial_poll_interval_ms = 1;  // Start with very short intervals
      int poll_count = 0;
      bool data_ready = false;
      int backoff_factor = 1;

      while (poll_count < max_poll_attempts && !data_ready) {
        // Check if measurements are available
        if (cupti_profiler_->hasMeasurements()) {
          data_ready = true;
          break;
        }

        // If no pending launches, the measurement was likely lost
        if (!cupti_profiler_->hasPendingLaunches()) {
          break;
        }

        // Adaptive sleep with exponential backoff (but capped)
        int current_interval = std::min(initial_poll_interval_ms * backoff_factor, 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(current_interval));

        // More frequent flushes during high contention
        if (poll_count % 5 == 0) {
          cuptiActivityFlushAll(0);
        }

        // Increase backoff every 50 attempts, but cap at 5ms
        if (poll_count % 50 == 0 && backoff_factor < 5) {
          backoff_factor++;
        }

        poll_count++;
      }

      cuda_kernel_launch_start_time = cupti_profiler_->getLatestSchedulingLatency();
      cuda_kernel_execution_time = cupti_profiler_->getLatestExecutionDuration();

      // More detailed timeout warning
      if (poll_count >= max_poll_attempts) {
        HOLOSCAN_LOG_WARN("[CUPTI] Data polling timed out after {} attempts (~{}ms). "
                          "Possible buffer overflow or severe contention.", poll_count,
                          (poll_count * 2));
      }
    } else {
      cuda_kernel_launch_start_time = -1.0;  // CUPTI not available
      HOLOSCAN_LOG_WARN("[TimingBenchmarkOp] CUPTI profiler not available!");
    }

    cuda_kernel_launch_start_times_us_.push_back(cuda_kernel_launch_start_time);
    cuda_kernel_execution_times_us_.push_back(cuda_kernel_execution_time);

    // Stop the entire benchmark app if the total number of samples is reached
    if (execution_count_ >= total_samples_.get()) {
      fragment()->stop_execution();
    }
  }

  BenchmarkStats get_cuda_kernel_launch_start_time_benchmark_stats() const {
    return calculate_benchmark_stats(cuda_kernel_launch_start_times_us_, true);
  }

  BenchmarkStats get_cuda_kernel_execution_time_benchmark_stats() const {
    return calculate_benchmark_stats(cuda_kernel_execution_times_us_, true);
  }

  ~TimingBenchmarkOp() {
    if (d_benchmark_data_) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(d_benchmark_data_), "cudaFree failed");
    }
  }

 private:
  Parameter<int> workload_size_;
  Parameter<int> total_samples_;
  int execution_count_ = 0;
  float* d_benchmark_data_;
  std::vector<double> cuda_kernel_launch_start_times_us_;
  std::vector<double> cuda_kernel_execution_times_us_;
  cudaStream_t cached_stream_ = nullptr;
  cupti_timing::CuptiSchedulingProfiler* cupti_profiler_;
  // CUDA kernel parameters
  int threads_per_block_;
  int blocks_;
};


class GreenContextBenchmarkApp : public holoscan::Application {
 public:
  explicit GreenContextBenchmarkApp(bool use_green_context = false, int total_samples = 100,
                                    int background_load_intensity = 1000,
                                    int background_load_size = 2097152, int threads_per_block = 512)
      : use_green_context_(use_green_context),
        total_samples_(total_samples),
        background_load_intensity_(background_load_intensity),
        background_load_size_(background_load_size),
        threads_per_block_(threads_per_block) {}

  void compose() override {
    dummy_load_op_ = make_operator<DummyLoadOp>(
        "dummy_load_op",
        Arg("load_intensity", background_load_intensity_),
        Arg("workload_size", background_load_size_),
        Arg("threads_per_block", threads_per_block_));
    add_operator(dummy_load_op_);

    timing_benchmark_op_ = make_operator<TimingBenchmarkOp>(
        "timing_benchmark_op",
        Arg("workload_size", 1024),
        Arg("total_samples", total_samples_));
    add_operator(timing_benchmark_op_);

    if (use_green_context_) {
      HOLOSCAN_LOG_INFO("Initializing green context partitions");
      try {
        // Check GPU properties first
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        HOLOSCAN_LOG_INFO("GPU: {}, SMs: {}, Compute: {}.{}", prop.name, prop.multiProcessorCount,
          prop.major, prop.minor);

        // Create separate Green Context partitions for proper isolation testing
        // Green Context requires partition size to be multiple of min_sm_count*2 = 4
        int total_sms = prop.multiProcessorCount;
        int sms_per_partition = std::max(4, (total_sms / 2) & ~3);  // Use half total SMs, minimum 4

        // Verify GPU has enough SMs for partitioning
        if (sms_per_partition * 2 > total_sms) {
          throw std::runtime_error("GPU too small for Green Context (need at least 8 SMs)");
        }

        std::vector<uint32_t> partitions = {static_cast<uint32_t>(sms_per_partition),
                                            static_cast<uint32_t>(sms_per_partition)};

        HOLOSCAN_LOG_INFO("Configuring green context with {} partitions, {} SMs each "
                          "(total: {} SMs, available: {} SMs)", partitions.size(),
                          sms_per_partition, sms_per_partition * 2, total_sms);

        auto cuda_green_context_pool = make_resource<CudaGreenContextPool>(
            "cuda_green_context_pool",
            Arg("dev_id", 0),
            Arg("num_partitions", static_cast<uint32_t>(partitions.size())),
            Arg("sms_per_partition", partitions));

        // Green Context Partition 1 - for DummyLoadOp
        auto dummy_green_context =
            make_resource<CudaGreenContext>("dummy_green_context",
                                            Arg("cuda_green_context_pool", cuda_green_context_pool),
                                            Arg("index", static_cast<int32_t>(0)));

        auto dummy_stream_pool =
            make_resource<CudaStreamPool>("dummy_stream_pool", 0, 0, 0, 1, 5, dummy_green_context);

        // Green Context Partition 2 - for TimingBenchmarkOp
        auto timing_green_context =
            make_resource<CudaGreenContext>("timing_green_context",
                                            Arg("cuda_green_context_pool", cuda_green_context_pool),
                                            Arg("index", static_cast<int32_t>(1)));

        auto timing_stream_pool = make_resource<CudaStreamPool>(
            "timing_stream_pool", 0, 0, 0, 1, 5, timing_green_context);

        dummy_load_op_->add_arg(dummy_stream_pool);
        dummy_load_op_->add_arg(dummy_green_context);
        dummy_load_op_->add_arg(cuda_green_context_pool);

        timing_benchmark_op_->add_arg(timing_stream_pool);
        timing_benchmark_op_->add_arg(timing_green_context);
        timing_benchmark_op_->add_arg(cuda_green_context_pool);

        HOLOSCAN_LOG_INFO("Green context enabled with separate partitions:");
        HOLOSCAN_LOG_INFO("  - DummyLoadOp: Partition 0 ({}) SMs", sms_per_partition);
        HOLOSCAN_LOG_INFO("  - TimingBenchmarkOp: Partition 1 ({}) SMs", sms_per_partition);
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Failed to setup green context: {}", e.what());
        throw;  // Re-throw the caught exception
      }
    } else {
      // Baseline: Both operators use non-default streams but NO Green Context partitions
      try {
        // Create regular (non-Green Context) stream pools for both operators
        auto dummy_stream_pool = make_resource<CudaStreamPool>("dummy_stream_pool", 0, 0, 0, 1, 5);

        auto timing_stream_pool =
            make_resource<CudaStreamPool>("timing_stream_pool", 0, 0, 0, 1, 5);

        dummy_load_op_->add_arg(dummy_stream_pool);
        timing_benchmark_op_->add_arg(timing_stream_pool);

        HOLOSCAN_LOG_INFO("Baseline mode: Non-default streams WITHOUT Green Context partitions");
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Failed to setup baseline stream pools: {}", e.what());
        throw;  // Re-throw the caught exception
      }
    }
  }

  BenchmarkStats get_cuda_kernel_launch_start_time_benchmark_stats() const {
    return timing_benchmark_op_->get_cuda_kernel_launch_start_time_benchmark_stats();
  }

  BenchmarkStats get_cuda_kernel_execution_time_benchmark_stats() const {
    return timing_benchmark_op_->get_cuda_kernel_execution_time_benchmark_stats();
  }

  BenchmarkStats get_dummy_load_execution_time_benchmark_stats() const {
    return dummy_load_op_->get_compute_execution_time_benchmark_stats();
  }

 private:
  bool use_green_context_;
  int total_samples_;
  int background_load_intensity_;
  int background_load_size_;      // Number of elements for DummyLoadOp GPU kernel
  int threads_per_block_;  // CUDA threads per block for DummyLoadOp kernel
  std::shared_ptr<DummyLoadOp> dummy_load_op_;
  std::shared_ptr<TimingBenchmarkOp> timing_benchmark_op_;
};

void print_comprehensive_timing_results(const BenchmarkStats& launch_start_stats,
                                       const BenchmarkStats& execution_stats,
                                       const std::string& context_type) {
  std::cout << "=== " << context_type << " ===" << std::endl;
  std::cout << std::fixed << std::setprecision(2) << std::dec;

  // Launch-Start Time Results
  std::cout << "CUDA Kernel Launch-Start Time:" << std::endl;
  if (launch_start_stats.sample_count == 0) {
    std::cout << "  Not available" << std::endl;
    std::cout << "  (CUPTI initialization may have failed or no measurements captured)"
    << std::endl;
  } else {
    std::cout << "  Average: " << launch_start_stats.avg << " μs" << std::endl;
    std::cout << "  Std Dev: " << launch_start_stats.std_dev << " μs" << std::endl;
    std::cout << "  Min:     " << launch_start_stats.min_val << " μs" << std::endl;
    std::cout << "  P50:     " << launch_start_stats.p50 << " μs" << std::endl;
    std::cout << "  P95:     " << launch_start_stats.p95 << " μs" << std::endl;
    std::cout << "  P99:     " << launch_start_stats.p99 << " μs" << std::endl;
    std::cout << "  Max:     " << launch_start_stats.max_val << " μs" << std::endl;
    std::cout << "  Samples: " << launch_start_stats.sample_count << std::endl;
  }

  std::cout << std::endl;

  // Kernel Execution Time Results
  std::cout << "CUDA Kernel Execution Time:" << std::endl;
  if (execution_stats.sample_count == 0) {
    std::cout << "  Not available" << std::endl;
    std::cout << "  (CUPTI initialization may have failed or no measurements captured)"
    << std::endl;
  } else {
    std::cout << "  Average: " << execution_stats.avg << " μs" << std::endl;
    std::cout << "  Std Dev: " << execution_stats.std_dev << " μs" << std::endl;
    std::cout << "  Min:     " << execution_stats.min_val << " μs" << std::endl;
    std::cout << "  P50:     " << execution_stats.p50 << " μs" << std::endl;
    std::cout << "  P95:     " << execution_stats.p95 << " μs" << std::endl;
    std::cout << "  P99:     " << execution_stats.p99 << " μs" << std::endl;
    std::cout << "  Max:     " << execution_stats.max_val << " μs" << std::endl;
    std::cout << "  Samples: " << execution_stats.sample_count << std::endl;
  }
}

void print_title(const std::string& title) {
  std::cout << std::string(80, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(80, '=') << std::endl;
}

void print_benchmark_config(const std::string& mode, int total_samples,
                            int background_load_intensity, int background_load_size_mb,
                            int background_load_size, int threads_per_block) {
  std::cout << "  Benchmark Mode: " << mode << std::endl;
  std::cout << "  Measurement Samples: " << total_samples << std::endl;
  std::cout << "  Background Load Intensity: " << background_load_intensity << std::endl;
  std::cout << "  Background Load Size: " << background_load_size_mb << " MB ("
            << background_load_size << " elements)" << std::endl;
  std::cout << "  CUDA Threads Per Block: " << threads_per_block << std::endl;
}

void print_usage(const char* program_name) {
  std::cout << "Green Context CUDA Kernel Launch-Start Time Benchmark\n\n";
  std::cout
      << "Measures time from CUDA kernel launch to execution start\n\n";
  std::cout << "Usage: " << program_name << " [OPTIONS]\n";
  std::cout << "Options:\n";
  std::cout << "  --samples N        Number of timing samples to measure (default: 1000)\n";
  std::cout << "  --load-intensity N    GPU load intensity multiplier (default: 20)\n";
  std::cout << "  --workload-size N     GPU memory size in MB for DummyLoadOp "
               "(default: 8)\n";
  std::cout << "  --threads-per-block N CUDA threads per block for GPU kernels "
               "(default: 512)\n";
  std::cout << "  --mode MODE          Run mode: 'baseline', 'green-context', "
               "or 'all' (default: all)\n";
  std::cout << "                        baseline: Run only without green context\n";
  std::cout << "                        green-context: Run only with green context\n";
  std::cout << "                        all: Run both and show comparison\n";
  std::cout << "  --help               Show this help message\n";
  std::cout << "\nExample:\n";
  std::cout << "  " << program_name
            << " --samples 1000 --load-intensity 10 --workload-size "
               "8 --threads-per-block 512 --mode all\n";
  std::cout << "\nNote: Requires CUPTI. May need admin privileges or driver configuration.\n";
}

int main(int argc, char* argv[]) {
  // Default values
  int total_samples = 1000;
  int background_load_intensity = 20;
  int background_load_size_mb = 8;     // GPU memory size in MB for DummyLoadOp
  int threads_per_block = 512;  // CUDA threads per block for GPU kernels
  std::string mode = "all";     // Run mode: "baseline", "green-context", or "all"

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--samples" && i + 1 < argc) {
      total_samples = std::atoi(argv[++i]);
      if (total_samples <= 0) {
        std::cerr << "Error: samples must be positive\n";
        return 1;
      }
    } else if (arg == "--load-intensity" && i + 1 < argc) {
      background_load_intensity = std::atoi(argv[++i]);
      if (background_load_intensity <= 0) {
        std::cerr << "Error: load-intensity must be positive\n";
        return 1;
      }
    } else if (arg == "--workload-size" && i + 1 < argc) {
      background_load_size_mb = std::atoi(argv[++i]);
      if (background_load_size_mb <= 0) {
        std::cerr << "Error: workload-size must be positive\n";
        return 1;
      }
    } else if (arg == "--threads-per-block" && i + 1 < argc) {
      threads_per_block = std::atoi(argv[++i]);
      if (threads_per_block <= 0) {
        std::cerr << "Error: threads-per-block must be positive\n";
        return 1;
      }
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
      if (mode != "baseline" && mode != "green-context" && mode != "all") {
        std::cerr << "Error: mode must be 'baseline', 'green-context', or 'all'\n";
        return 1;
      }
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  // Convert workload size from MB to number of float elements
  int background_load_size = (background_load_size_mb * 1024 * 1024) / sizeof(float);

  print_title("Green Context CUDA Kernel Start Time Benchmark");
  std::cout << "Benchmark Configurations:" << std::endl;
  print_benchmark_config(mode, total_samples, background_load_intensity,
                        background_load_size_mb, background_load_size, threads_per_block);

  // Initialize CUDA
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(0), "Failed to set CUDA device");

  // Initialize CUPTI profiler
  std::cout << "\nInitializing CUPTI profiler..." << std::endl;
  auto* cupti_profiler = cupti_timing::CuptiSchedulingProfiler::getInstance();
  if (!cupti_profiler->initialize()) {
    std::cout << "[WARNING] CUPTI initialization failed - will use host-side timing only"
              << std::endl;
  }

  BenchmarkStats cuda_kernel_launch_start_time_stats_without_gc,
                 cuda_kernel_launch_start_time_stats_with_gc;
  BenchmarkStats dummy_load_stats_without_gc, dummy_load_stats_with_gc;
  BenchmarkStats cuda_kernel_execution_stats_without_gc, cuda_kernel_execution_stats_with_gc;

  // Run benchmark without green context (baseline: both kernels on separate non-default streams)
  if (mode == "baseline" || mode == "all") {
    print_title("Running benchmark for baseline\n"
                "(non-default CUDA streams, without green context)");

    try {
      auto app_no_gc = std::make_unique<GreenContextBenchmarkApp>(
          false, total_samples, background_load_intensity, background_load_size, threads_per_block);
      app_no_gc->scheduler(app_no_gc->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));
      app_no_gc->run();
      cuda_kernel_launch_start_time_stats_without_gc =
          app_no_gc->get_cuda_kernel_launch_start_time_benchmark_stats();
      dummy_load_stats_without_gc =
          app_no_gc->get_dummy_load_execution_time_benchmark_stats();
      cuda_kernel_execution_stats_without_gc =
          app_no_gc->get_cuda_kernel_execution_time_benchmark_stats();
      std::cout << "Baseline benchmark completed" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Baseline benchmark failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Run benchmark with green context enabled (separate partitions for each kernel)
  if (mode == "green-context" || mode == "all") {
    print_title("Running main benchmark\n"
                "(with green context, separate partitions for each kernel)");

    try {
      auto app_with_gc = std::make_unique<GreenContextBenchmarkApp>(
          true, total_samples, background_load_intensity, background_load_size, threads_per_block);
      app_with_gc->scheduler(app_with_gc->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based", holoscan::Arg("worker_thread_number", static_cast<int64_t>(6))));
      app_with_gc->run();
      cuda_kernel_launch_start_time_stats_with_gc =
          app_with_gc->get_cuda_kernel_launch_start_time_benchmark_stats();
      dummy_load_stats_with_gc =
          app_with_gc->get_dummy_load_execution_time_benchmark_stats();
      cuda_kernel_execution_stats_with_gc =
          app_with_gc->get_cuda_kernel_execution_time_benchmark_stats();
      std::cout << "Main benchmark completed" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Main benchmark failed: " << e.what() << std::endl;
      return 1;
    }
  }

  // Display benchmark configurations
  print_title("Benchmark Configurations");
  print_benchmark_config(mode, total_samples, background_load_intensity,
                         background_load_size_mb, background_load_size, threads_per_block);
  std::cout << std::endl;

  // Display comprehensive benchmark results (launch-start time + execution time)
  print_title("Comprehensive Timing Results");

  if (mode == "baseline" || mode == "all") {
    print_comprehensive_timing_results(cuda_kernel_launch_start_time_stats_without_gc,
                                      cuda_kernel_execution_stats_without_gc,
                                      "Without Green Context (Baseline)");
    std::cout << std::endl;
  }

  if (mode == "green-context" || mode == "all") {
    print_comprehensive_timing_results(cuda_kernel_launch_start_time_stats_with_gc,
                                      cuda_kernel_execution_stats_with_gc,
                                      "With Green Context");
    std::cout << std::endl;
  }

  // Performance Comparison - only show when mode is "all"
  if (mode == "all" && cuda_kernel_launch_start_time_stats_without_gc.sample_count > 0 &&
      cuda_kernel_launch_start_time_stats_with_gc.sample_count > 0) {
    print_title("Baseline and Green Context Benchmark Comparison");

    // Calculate improvements in launch-start latency
    double avg_improvement = ((cuda_kernel_launch_start_time_stats_without_gc.avg -
                               cuda_kernel_launch_start_time_stats_with_gc.avg) /
                              cuda_kernel_launch_start_time_stats_without_gc.avg) * 100.0;
    double p95_improvement = ((cuda_kernel_launch_start_time_stats_without_gc.p95 -
                               cuda_kernel_launch_start_time_stats_with_gc.p95) /
                              cuda_kernel_launch_start_time_stats_without_gc.p95) * 100.0;
    double p99_improvement = ((cuda_kernel_launch_start_time_stats_without_gc.p99 -
                               cuda_kernel_launch_start_time_stats_with_gc.p99) /
                              cuda_kernel_launch_start_time_stats_without_gc.p99) * 100.0;

    std::cout << std::fixed << std::setprecision(2) << std::dec;

    std::cout << "Launch-Start Latency:" << std::endl;
    std::cout << "  Average Latency:  " << std::setw(8)
              << cuda_kernel_launch_start_time_stats_without_gc.avg << " μs → "
              << std::setw(8) << cuda_kernel_launch_start_time_stats_with_gc.avg
              << " μs  (" << std::showpos << avg_improvement << "%)"
              << std::endl;
    std::cout << "  95th Percentile:  " << std::setw(8)
              << cuda_kernel_launch_start_time_stats_without_gc.p95 << " μs → "
              << std::setw(8) << cuda_kernel_launch_start_time_stats_with_gc.p95
              << " μs  (" << std::showpos << p95_improvement << "%)"
              << std::endl;
    std::cout << "  99th Percentile:  " << std::setw(8)
              << cuda_kernel_launch_start_time_stats_without_gc.p99 << " μs → "
              << std::setw(8) << cuda_kernel_launch_start_time_stats_with_gc.p99
              << " μs  (" << std::showpos << p99_improvement << "%)"
              << std::endl << std::endl;
    std::cout << std::noshowpos;

    // Add kernel execution time comparison if data is available
    if (cuda_kernel_execution_stats_without_gc.sample_count > 0 &&
        cuda_kernel_execution_stats_with_gc.sample_count > 0) {
      // Calculate improvements in kernel execution time
      double exec_avg_improvement = ((cuda_kernel_execution_stats_without_gc.avg -
                                     cuda_kernel_execution_stats_with_gc.avg) /
                                    cuda_kernel_execution_stats_without_gc.avg) * 100.0;
      double exec_p95_improvement = ((cuda_kernel_execution_stats_without_gc.p95 -
                                     cuda_kernel_execution_stats_with_gc.p95) /
                                    cuda_kernel_execution_stats_without_gc.p95) * 100.0;
      double exec_p99_improvement = ((cuda_kernel_execution_stats_without_gc.p99 -
                                     cuda_kernel_execution_stats_with_gc.p99) /
                                    cuda_kernel_execution_stats_without_gc.p99) * 100.0;

      std::cout << "Kernel Execution Time:" << std::endl;
      std::cout << "  Average Duration: " << std::setw(8)
                << cuda_kernel_execution_stats_without_gc.avg << " μs → "
                << std::setw(8) << cuda_kernel_execution_stats_with_gc.avg
                << " μs  (" << std::showpos << exec_avg_improvement << "%)"
                << std::endl;
      std::cout << "  95th Percentile:  " << std::setw(8)
                << cuda_kernel_execution_stats_without_gc.p95 << " μs → "
                << std::setw(8) << cuda_kernel_execution_stats_with_gc.p95
                << " μs  (" << std::showpos << exec_p95_improvement << "%)"
                << std::endl;
      std::cout << "  99th Percentile:  " << std::setw(8)
                << cuda_kernel_execution_stats_without_gc.p99 << " μs → "
                << std::setw(8) << cuda_kernel_execution_stats_with_gc.p99
                << " μs  (" << std::showpos << exec_p99_improvement << "%)"
                << std::endl << std::endl;
      std::cout << std::noshowpos;
    }
  }

  // Display DummyLoadOp execution time statistics
  print_title("Dummy Load Execution Time Statistics");

  if (mode == "baseline" || mode == "all") {
    if (dummy_load_stats_without_gc.sample_count > 0) {
      std::cout << "=== Without Green Context (Baseline) ===" << std::endl;
      std::cout << std::fixed << std::setprecision(2);
      std::cout << "  Average: " << dummy_load_stats_without_gc.avg << " μs" << std::endl;
      std::cout << "  Std Dev: " << dummy_load_stats_without_gc.std_dev << " μs" << std::endl;
      std::cout << "  Samples: " << dummy_load_stats_without_gc.sample_count << std::endl
                << std::endl;
    }
  }

  if (mode == "green-context" || mode == "all") {
    if (dummy_load_stats_with_gc.sample_count > 0) {
      std::cout << "=== With Green Context ===" << std::endl;
      std::cout << std::fixed << std::setprecision(2);
      std::cout << "  Average: " << dummy_load_stats_with_gc.avg << " μs" << std::endl;
      std::cout << "  Std Dev: " << dummy_load_stats_with_gc.std_dev << " μs" << std::endl;
      std::cout << "  Samples: " << dummy_load_stats_with_gc.sample_count << std::endl << std::endl;
    }
  }

  // Cleanup CUPTI profiler
  if (cupti_profiler) {
    cupti_profiler->cleanup();
  }

  return 0;
}
