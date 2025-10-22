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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <getopt.h>
#include <unistd.h>

#include <holoscan/holoscan.hpp>

using namespace holoscan;

struct BenchmarkStats {
  double avg = 0.0;
  double std_dev = 0.0;
  double min_val = 0.0;
  double max_val = 0.0;
  size_t sample_count = 0;
  std::vector<double> raw_data;
  std::vector<double> sorted_data;
};


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
      stats.raw_data.push_back(value);
    }
  }

  if (stats.raw_data.empty())
    return stats;

  stats.sorted_data = stats.raw_data;
  std::sort(stats.sorted_data.begin(), stats.sorted_data.end());
  stats.sample_count = stats.sorted_data.size();

  // Calculate basic statistics
  stats.avg =
      std::accumulate(stats.sorted_data.begin(), stats.sorted_data.end(), 0.0) / stats.sample_count;
  stats.std_dev = calculate_std_dev(stats.sorted_data, stats.avg);

  // Calculate min/max
  stats.min_val = stats.sorted_data.front();
  stats.max_val = stats.sorted_data.back();

  return stats;
}


static double run_dummy_cpu_workload(int workload_size = 100, int load_intensity = 100) {
  std::vector<double> data(workload_size);
  double work_result = 0.0;
  for (int i = 0; i < load_intensity; ++i) {
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = std::sin(i * 0.01) * std::cos(i * 0.02);
    }
    for (double x : data) {
      work_result += std::sqrt(std::abs(x) + 1.0);
    }
  }
  return work_result;
}

class DummyLoadOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyLoadOp)

  explicit DummyLoadOp(int workload_size = 1000, int load_intensity = 100)
  : workload_size_(workload_size), load_intensity_(load_intensity) {}

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    sink_ += run_dummy_cpu_workload(workload_size_, load_intensity_);
  };

 private:
  int workload_size_;
  int load_intensity_;
  double sink_ = 0.0;  // prevents optimization
};

class BenchmarkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BenchmarkOp)

  BenchmarkOp(int target_fps = 60,
              int load_intensity = 100,
              int workload_size = 100)
            : target_fps_(target_fps),
              load_intensity_(load_intensity),
              workload_size_(workload_size) {}

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto start_time = std::chrono::steady_clock::now();

    if (last_start_time_ == std::chrono::steady_clock::time_point()) {
      last_start_time_ = start_time;
    } else {
      auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        start_time - last_start_time_).count();
      periods_ns_.push_back(period_ns);
      last_start_time_ = start_time;
    }

    work_result_ += run_dummy_cpu_workload(workload_size_, load_intensity_);
    auto end_time = std::chrono::steady_clock::now();

    auto execution_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_time - start_time).count();
    execution_times_ns_.push_back(execution_time_ns);
  };

  BenchmarkStats get_execution_time_benchmark_stats() const {
    BenchmarkStats benchmark_stats;

    if (execution_times_ns_.empty()) {
      return benchmark_stats;
    }

    // Convert nanoseconds to milliseconds for statistics
    std::vector<double> execution_times_ms;
    for (double ns : execution_times_ns_) {
      execution_times_ms.push_back(ns / 1e6);
    }

    benchmark_stats = calculate_benchmark_stats(execution_times_ms, false);

    return benchmark_stats;
  }

  BenchmarkStats get_period_benchmark_stats() const {
    BenchmarkStats benchmark_stats;
    if (periods_ns_.empty()) {
      return benchmark_stats;
    }

    // Convert nanoseconds to milliseconds for statistics
    std::vector<double> periods_ms;
    for (double ns : periods_ns_) {
      periods_ms.push_back(ns / 1e6);
    }

    benchmark_stats = calculate_benchmark_stats(periods_ms, false);
    return benchmark_stats;
  }

 private:
  int target_fps_;
  int load_intensity_;
  int workload_size_;
  std::vector<double> periods_ns_;
  std::vector<double> execution_times_ns_;
  std::chrono::steady_clock::time_point last_start_time_{};
  mutable std::mutex lock_;
  double work_result_ = 0.0;
};

class RealtimeThreadBenchmarkApp : public Application {
 public:
  RealtimeThreadBenchmarkApp(int target_fps = 60,
                         bool use_realtime = false,
                         SchedulingPolicy scheduling_policy = SchedulingPolicy::kDeadline,
                         int bg_load_intensity = 1000,
                         int bg_workload_size = 100,
                         int bm_load_intensity = 100,
                         int bm_workload_size = 100,
                         int dummy_load_number = 2,
                         unsigned int pin_cpu = 0)
      : target_fps_(target_fps),
        use_realtime_(use_realtime),
        scheduling_policy_(scheduling_policy),
        bg_load_intensity_(bg_load_intensity),
        bg_workload_size_(bg_workload_size),
        bm_load_intensity_(bm_load_intensity),
        bm_workload_size_(bm_workload_size),
        dummy_load_number_(dummy_load_number),
        pin_cpu_(pin_cpu) {}

  void compose() override {
    benchmark_op_ = make_operator<BenchmarkOp>("benchmark_op", target_fps_,
                                               bm_load_intensity_, bm_workload_size_);
    add_operator(benchmark_op_);

    auto periodic_condition = make_condition<PeriodicCondition>("periodic_condition",
                                                                1000000000 / target_fps_);

    if (use_realtime_) {
      // Create a thread pool for hosting the real-time thread
      auto realtime_pool = make_thread_pool("realtime_thread_pool");

      if (scheduling_policy_ == SchedulingPolicy::kDeadline) {
        int64_t period_ns = static_cast<int64_t>(1e9 / target_fps_);
        int64_t deadline_ns = period_ns;
        int64_t runtime_ns = static_cast<int64_t>(period_ns * 0.10);  // 10% of period

        realtime_pool->add_realtime(benchmark_op_, scheduling_policy_, true, {pin_cpu_}, 0,
                                   runtime_ns, deadline_ns, period_ns);
      } else if (scheduling_policy_ == SchedulingPolicy::kFirstInFirstOut ||
                 scheduling_policy_ == SchedulingPolicy::kRoundRobin) {
        realtime_pool->add_realtime(benchmark_op_, scheduling_policy_, true, {pin_cpu_}, 99);
        benchmark_op_->add_arg(periodic_condition);
      }
    } else {
      benchmark_op_->add_arg(periodic_condition);
    }

    // Create dummy load operators based on dummy_load_number_
    for (int i = 0; i < dummy_load_number_; ++i) {
      auto dummy_load_op = make_operator<DummyLoadOp>(
        ("dummy_load_op_" + std::to_string(i + 1)), bg_workload_size_, bg_load_intensity_);
      add_operator(dummy_load_op);
    }
  }

  BenchmarkStats get_execution_time_benchmark_stats() {
    return benchmark_op_->get_execution_time_benchmark_stats();
  }

  BenchmarkStats get_period_benchmark_stats() {
    return benchmark_op_->get_period_benchmark_stats();
  }

 private:
  int target_fps_;
  bool use_realtime_;
  SchedulingPolicy scheduling_policy_;
  int bg_load_intensity_;
  int bg_workload_size_;
  int bm_load_intensity_;
  int bm_workload_size_;
  int dummy_load_number_;
  unsigned int pin_cpu_;
  std::shared_ptr<BenchmarkOp> benchmark_op_;
};

void print_title(const std::string& title) {
  std::cout << std::string(80, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(80, '=') << std::endl;
}

void print_benchmark_config(int target_fps, int duration_seconds,
                           const std::string& scheduling_policy_str,
                           int bg_load_intensity, int bg_workload_size,
                           int bm_load_intensity, int bm_workload_size,
                           bool use_realtime, int worker_thread_number,
                           int dummy_load_number) {
  double target_period_ms = 1000.0 / target_fps;
  std::cout << "  Target FPS: " << target_fps << " (" << std::fixed << std::setprecision(3)
            << target_period_ms << " ms period)" << std::endl;
  std::cout << "  Duration: " << duration_seconds << "s" << std::endl;
  std::cout << "  Realtime: " << (use_realtime ? "true" : "false") << std::endl;
  if (use_realtime) {
    std::cout << "  Scheduling Policy: " << scheduling_policy_str << std::endl;
  }
  std::cout << "  Background Load Intensity: " << bg_load_intensity << std::endl;
  std::cout << "  Background Workload Size: " << bg_workload_size << std::endl;
  std::cout << "  Benchmark Load Intensity: " << bm_load_intensity << std::endl;
  std::cout << "  Benchmark Workload Size: " << bm_workload_size << std::endl;
  std::cout << "  Worker Thread Number: " << worker_thread_number << std::endl;
  std::cout << "  Dummy Load Number: " << dummy_load_number << std::endl;
}

void print_benchmark_results(
  const BenchmarkStats& period_stats,
  const BenchmarkStats& execution_time_stats,
  int target_fps,
  const std::string& context_type) {
  std::cout << "=== " << context_type << " ===" << std::endl;
  std::cout << std::fixed << std::setprecision(6) << std::dec;

  if (period_stats.sample_count > 0) {
    std::cout << "Frame period std: " << period_stats.std_dev << " ms" << std::endl;
    std::cout << "Frame period mean: " << period_stats.avg << " ms" << std::endl;
    std::cout << "Frame period min/max: " << period_stats.min_val << " ms / "
              << period_stats.max_val << " ms" << std::endl << std::endl;
  }
}

void write_stats_section(std::ofstream& file, const BenchmarkStats& stats) {
  file << "      \"raw_data\": [";
  for (size_t i = 0; i < stats.raw_data.size(); ++i) {
    if (i > 0) file << ", ";
    file << stats.raw_data[i];
  }
  file << "],\n";
  file << "      \"statistics\": {\n";
  file << "        \"sample_count\": " << stats.sample_count << ",\n";
  file << "        \"average\": " << stats.avg << ",\n";
  file << "        \"std_dev\": " << stats.std_dev << ",\n";
  file << "        \"min\": " << stats.min_val << ",\n";
  file << "        \"max\": " << stats.max_val << "\n";
  file << "      }\n";
}

void write_json_results(const std::string& filename,
  const BenchmarkStats& non_rt_period_stats,
  const BenchmarkStats& rt_period_stats,
  const BenchmarkStats& non_rt_execution_stats,
  const BenchmarkStats& rt_execution_stats,
  int target_fps,
  int duration_seconds,
  const std::string& scheduling_policy_str,
  int bg_load_intensity,
  int bg_workload_size,
  int bm_load_intensity,
  int bm_workload_size,
  int worker_thread_number,
  int dummy_load_number) {
  try {
    std::filesystem::path p{filename};
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: Could not create output directory for " << filename << ": " << e.what()
              << std::endl;
    return;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
    return;
  }

  file << std::fixed << std::setprecision(6);
  file << "{\n";
  file << "  \"benchmark_config\": {\n";
  file << "    \"target_fps\": " << target_fps << ",\n";
  file << "    \"duration_seconds\": " << duration_seconds << ",\n";
  file << "    \"scheduling_policy\": \"" << scheduling_policy_str << "\",\n";
  file << "    \"bg_load_intensity\": " << bg_load_intensity << ",\n";
  file << "    \"bg_workload_size\": " << bg_workload_size << ",\n";
  file << "    \"bm_load_intensity\": " << bm_load_intensity << ",\n";
  file << "    \"bm_workload_size\": " << bm_workload_size << ",\n";
  file << "    \"worker_thread_number\": " << worker_thread_number << ",\n";
  file << "    \"dummy_load_number\": " << dummy_load_number << "\n";
  file << "  },\n";

  file << "  \"period_statistics\": {\n";
  file << "    \"non_realtime\": {\n";
  write_stats_section(file, non_rt_period_stats);
  file << "    },\n";
  file << "    \"realtime\": {\n";
  write_stats_section(file, rt_period_stats);
  file << "    }\n";
  file << "  },\n";

  file << "  \"execution_time_statistics\": {\n";
  file << "    \"non_realtime\": {\n";
  write_stats_section(file, non_rt_execution_stats);
  file << "    },\n";
  file << "    \"realtime\": {\n";
  write_stats_section(file, rt_execution_stats);
  file << "    }\n";
  file << "  }\n";
  file << "}\n";

  file.close();
  std::cout << "Raw measurement data written to: " << filename << std::endl;
}

void print_help(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]" << std::endl;
  std::cout << "\nOptions:" << std::endl;
  std::cout << "  -f, --target-fps <fps>          Target FPS (default: 60)" << std::endl;
  std::cout << "  -d, --duration <seconds>        Duration in seconds (default: 30)" << std::endl;
  std::cout << "  -p, --scheduling-policy <pol>   SCHED_DEADLINE, SCHED_FIFO, "
            << "or SCHED_RR (default: SCHED_DEADLINE)" << std::endl;
  std::cout << "  -I, --bg-load-intensity <int>   Background load intensity "
            << "(default: 1000)" << std::endl;
  std::cout << "  -S, --bg-workload-size <size>   Background workload size "
            << "(default: 100)" << std::endl;
  std::cout << "  -i, --bm-load-intensity <int>   Benchmark load intensity "
            << "(default: 100)" << std::endl;
  std::cout << "  -s, --bm-workload-size <size>   Benchmark workload size "
            << "(default: 100)" << std::endl;
  std::cout << "  -t, --worker-thread-number <n>  Worker thread number "
            << "(default: 2)" << std::endl;
  std::cout << "  -D, --dummy-load-number <num>   Dummy load operators "
            << "(default: 2)" << std::endl;
  std::cout << "  -c, --pin-cpu <cpu>             CPU to pin RT thread "
            << "(default: 0)" << std::endl;
  std::cout << "  -o, --output <file>             Output JSON file "
            << "(default: /tmp/benchmark_plots/...)" << std::endl;
  std::cout << "  -h, --help                      Show help message" << std::endl;
}

int main(int argc, char* argv[]) {
  // Default parameters
  int target_fps = 60;
  int duration_seconds = 30;
  std::string scheduling_policy_str = "SCHED_DEADLINE";
  int bg_load_intensity = 1000;
  int bg_workload_size = 100;
  int bm_load_intensity = 100;
  int bm_workload_size = 100;
  int worker_thread_number = 2;
  int dummy_load_number = 2;
  unsigned int pin_cpu = 0;
  std::string output_file = "/tmp/benchmark_plots/realtime_thread_benchmark_results.json";

  static struct option long_options[] = {
    {"target-fps", required_argument, nullptr, 'f'},
    {"duration", required_argument, nullptr, 'd'},
    {"scheduling-policy", required_argument, nullptr, 'p'},
    {"bg-load-intensity", required_argument, nullptr, 'I'},  // Background Intensity
    {"bg-workload-size", required_argument, nullptr, 'S'},   // Background Size
    {"bm-load-intensity", required_argument, nullptr, 'i'},  // benchmark intensity
    {"bm-workload-size", required_argument, nullptr, 's'},   // benchmark size
    {"worker-thread-number", required_argument, nullptr, 't'},
    {"dummy-load-number", required_argument, nullptr, 'D'},
    {"pin-cpu", required_argument, nullptr, 'c'},
    {"output", required_argument, nullptr, 'o'},
    {"help", no_argument, nullptr, 'h'},
    {nullptr, 0, nullptr, 0}
  };

  // Get number of CPUs configured on the system (Linux-specific)
  long num_cpus_long = sysconf(_SC_NPROCESSORS_CONF);
  unsigned int num_cpus;
  if (num_cpus_long <= 0) {
    num_cpus = 8;  // Fallback if detection fails
    std::cerr << "Warning: Could not detect CPU count, assuming "
              << num_cpus << " CPUs\n";
  } else {
    num_cpus = static_cast<unsigned int>(num_cpus_long);
  }

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "f:d:p:I:S:i:s:t:D:c:o:h",
                            long_options, &option_index)) != -1) {
    switch (opt) {
      case 'f':
        target_fps = std::atoi(optarg);
        if (target_fps <= 0) {
          std::cerr << "Error: target-fps must be positive\n";
          return 1;
        }
        break;
      case 'd':
        duration_seconds = std::atoi(optarg);
        if (duration_seconds <= 0) {
          std::cerr << "Error: duration must be positive\n";
          return 1;
        }
        break;
      case 'p':
        scheduling_policy_str = optarg;
        break;
      case 'I':
        bg_load_intensity = std::atoi(optarg);
        if (bg_load_intensity <= 0) {
          std::cerr << "Error: bg-load-intensity must be positive\n";
          return 1;
        }
        break;
      case 'S':
        bg_workload_size = std::atoi(optarg);
        if (bg_workload_size <= 0) {
          std::cerr << "Error: bg-workload-size must be positive\n";
          return 1;
        }
        break;
      case 'i':
        bm_load_intensity = std::atoi(optarg);
        if (bm_load_intensity <= 0) {
          std::cerr << "Error: bm-load-intensity must be positive\n";
          return 1;
        }
        break;
      case 's':
        bm_workload_size = std::atoi(optarg);
        if (bm_workload_size <= 0) {
          std::cerr << "Error: bm-workload-size must be positive\n";
          return 1;
        }
        break;
      case 't':
        worker_thread_number = std::atoi(optarg);
        if (worker_thread_number <= 1) {
          std::cerr << "Error: worker-thread-number must be greater than 1\n";
          return 1;
        }
        break;
      case 'D':
        dummy_load_number = std::atoi(optarg);
        if (dummy_load_number <= 0) {
          std::cerr << "Error: dummy-load-number must be positive\n";
          return 1;
        }
        break;
      case 'c': {
        int cpu_value = std::atoi(optarg);
        if (cpu_value < 0 || cpu_value >= static_cast<int>(num_cpus)) {
          std::cerr << "Error: pin-cpu must be between 0-" << (num_cpus - 1)
                    << " (system has " << num_cpus << " CPUs)\n";
          return 1;
        }
        pin_cpu = static_cast<unsigned int>(cpu_value);
        break;
      }
      case 'o':
        output_file = optarg;
        break;
      case 'h':
        print_help(argv[0]);
        return 0;
      case '?':
        print_help(argv[0]);
        return 1;
    }
  }

  // Map scheduling policy string to enum
  SchedulingPolicy scheduling_policy;
  if (scheduling_policy_str == "SCHED_DEADLINE") {
    scheduling_policy = SchedulingPolicy::kDeadline;
  } else if (scheduling_policy_str == "SCHED_FIFO") {
    scheduling_policy = SchedulingPolicy::kFirstInFirstOut;
  } else if (scheduling_policy_str == "SCHED_RR") {
    scheduling_policy = SchedulingPolicy::kRoundRobin;
  } else {
    std::cerr << "Invalid scheduling policy: " << scheduling_policy_str << std::endl;
    return 1;
  }

  print_title("Real-time Thread Benchmark");
  print_benchmark_config(target_fps, duration_seconds, scheduling_policy_str,
                          bg_load_intensity, bg_workload_size, bm_load_intensity,
                          bm_workload_size, false, worker_thread_number, dummy_load_number);

  // Run without real-time scheduling
  print_title("Running benchmark for baseline\n(without real-time thread)");
  auto non_rt_app = std::make_unique<RealtimeThreadBenchmarkApp>(
    target_fps, false, scheduling_policy, bg_load_intensity, bg_workload_size,
    bm_load_intensity, bm_workload_size, dummy_load_number, pin_cpu);
  non_rt_app->scheduler(non_rt_app->make_scheduler<EventBasedScheduler>(
    "event-based",
    Arg("worker_thread_number", static_cast<int64_t>(worker_thread_number)),
    Arg("max_duration_ms", static_cast<int64_t>(duration_seconds * 1000))));
  non_rt_app->run();
  auto non_rt_period_stats = non_rt_app->get_period_benchmark_stats();
  auto non_rt_execution_time_stats = non_rt_app->get_execution_time_benchmark_stats();

  // Run with real-time scheduling
  print_title("Running benchmark for real-time\n(with real-time scheduling)");
  std::cout << "  Pinning RT thread to CPU: " << pin_cpu << std::endl;
  auto rt_app = std::make_unique<RealtimeThreadBenchmarkApp>(
    target_fps, true, scheduling_policy, bg_load_intensity, bg_workload_size,
    bm_load_intensity, bm_workload_size, dummy_load_number, pin_cpu);
  rt_app->scheduler(rt_app->make_scheduler<EventBasedScheduler>(
    "event-based",
    Arg("worker_thread_number", static_cast<int64_t>(worker_thread_number)),
    Arg("max_duration_ms", static_cast<int64_t>(duration_seconds * 1000))));
  rt_app->run();
  auto rt_period_stats = rt_app->get_period_benchmark_stats();
  auto rt_execution_time_stats = rt_app->get_execution_time_benchmark_stats();

  // Display benchmark configurations
  print_title("Benchmark Configurations");
  print_benchmark_config(target_fps, duration_seconds, scheduling_policy_str,
                          bg_load_intensity, bg_workload_size, bm_load_intensity,
                          bm_workload_size, false, worker_thread_number, dummy_load_number);
  std::cout << std::endl;

  // Display benchmark results
  print_title("Benchmark Results");
  print_benchmark_results(non_rt_period_stats, non_rt_execution_time_stats, target_fps,
                           "Non-real-time Thread (Baseline)");
  print_benchmark_results(rt_period_stats, rt_execution_time_stats, target_fps,
                           "Real-time Thread");

  // Performance Comparison
  print_title("Non-real-time and Real-time Thread Benchmark Comparison");

  // Calculate improvement in standard deviation
  double std_improvement = 0.0;
  if (non_rt_period_stats.std_dev > 0.0) {
    std_improvement = ((non_rt_period_stats.std_dev - rt_period_stats.std_dev) /
                       non_rt_period_stats.std_dev) * 100.0;
  }

  std::cout << std::fixed << std::setprecision(6) << std::dec;

  std::cout << "Period std comparison: " << std::setw(8) << non_rt_period_stats.std_dev
            << " ms → " << std::setw(8) << rt_period_stats.std_dev << " ms  ("
            << std::showpos << std_improvement << "%)" << std::endl << std::endl;

  std::cout << std::noshowpos;

  // Write raw data to JSON file
  write_json_results(output_file,
    non_rt_period_stats,
    rt_period_stats,
    non_rt_execution_time_stats,
    rt_execution_time_stats,
    target_fps,
    duration_seconds,
    scheduling_policy_str,
    bg_load_intensity,
    bg_workload_size,
    bm_load_intensity,
    bm_workload_size,
    worker_thread_number,
    dummy_load_number);

  // Generate plots using Python plotting script
  std::string plot_script = std::string(__FILE__);
  plot_script = plot_script.substr(0, plot_script.find_last_of("/\\")) + "/plot_results.py";

  // Check if plot script exists
  std::ifstream script_file(plot_script);
  if (!script_file.good()) {
    std::cerr << "Error: Could not find plot script: " << plot_script << std::endl;
    std::cerr << "Skipping plot generation." << std::endl;
    return 0;
  }

  // Extract directory from JSON output file path for plots
  std::string output_dir = output_file.substr(0, output_file.find_last_of("/\\"));
  if (output_dir == output_file) {
    // No directory separator found, use current directory
    output_dir = ".";
  }

  std::string plot_command = "python3 " + plot_script + " --input " + output_file +
                             " --output-dir " + output_dir;

  std::cout << "\nGenerating plots..." << std::endl;
  int plot_result = std::system(plot_command.c_str());
  if (plot_result != 0) {
    std::cerr << "Warning: Failed to generate plots. "
              << "Make sure Python3 and matplotlib are installed" << std::endl;
  }

  return 0;
}
