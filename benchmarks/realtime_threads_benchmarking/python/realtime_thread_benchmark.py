# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
import statistics
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.resources import SchedulingPolicy
from holoscan.schedulers import EventBasedScheduler

matplotlib.use("Agg")  # Non-interactive backend for containers


class TargetOperator(Operator):
    """
    Target operator that aims to run at a specific FPS and measures timing performance.
    """

    def __init__(self, fragment, target_fps: int, *args, **kwargs):
        self.target_fps = target_fps
        self.target_period_ns = int(1e9 / target_fps)  # Period in nanoseconds
        self.frame_periods: list[float] = []  # Store frame periods in nanoseconds
        self.execution_times: list[float] = []  # Store execution times in nanoseconds
        self.frame_count = 0
        self.start_time: float | None = None
        self.last_frame_time: float | None = None
        self._lock = threading.Lock()

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def compute(self, op_input, op_output, context):
        current_time = time.time_ns()

        with self._lock:
            if self.start_time is None:
                # First frame - just initialize, don't calculate interval
                self.start_time = current_time
                self.last_frame_time = current_time
            else:
                # Calculate frame period for all frames after the first
                frame_period_ns = current_time - self.last_frame_time
                self.frame_periods.append(frame_period_ns)
                self.last_frame_time = current_time

            self.frame_count += 1

        self._do_real_work()
        execution_end = time.time_ns()

        with self._lock:
            execution_time_ns = execution_end - current_time
            self.execution_times.append(execution_time_ns)

    def get_statistics(self):
        """Get comprehensive performance statistics and raw data."""
        with self._lock:
            if len(self.frame_periods) < 1:
                return {
                    "target_fps": self.target_fps,
                    "frame_count": self.frame_count,
                    "total_duration_s": 0.0,
                    "frame_period_stats": {},
                    "execution_time_stats": {},
                    "frame_periods_ms": [],
                    "execution_times_ms": [],
                }

            # Calculate basic timing
            total_duration_ns = self.last_frame_time - self.start_time if self.start_time else 0
            total_duration_s = total_duration_ns / 1e9

            # Frame period statistics (convert to milliseconds)
            frame_periods_ms = [p / 1e6 for p in self.frame_periods]
            frame_period_stats = {}
            if frame_periods_ms:
                frame_period_stats = {
                    "mean_ms": statistics.mean(frame_periods_ms),
                    "std_ms": (
                        statistics.stdev(frame_periods_ms) if len(frame_periods_ms) > 1 else 0.0
                    ),
                    "min_ms": min(frame_periods_ms),
                    "max_ms": max(frame_periods_ms),
                }

            # Execution time statistics (convert to milliseconds)
            execution_times_ms = [t / 1e6 for t in self.execution_times]
            execution_time_stats = {}
            if execution_times_ms:
                execution_time_stats = {
                    "mean_ms": statistics.mean(execution_times_ms),
                    "std_ms": (
                        statistics.stdev(execution_times_ms) if len(execution_times_ms) > 1 else 0.0
                    ),
                    "min_ms": min(execution_times_ms),
                    "max_ms": max(execution_times_ms),
                }

            results = {
                "target_fps": self.target_fps,
                "frame_count": self.frame_count,
                "total_duration_s": total_duration_s,
                "frame_period_stats": frame_period_stats,
                "execution_time_stats": execution_time_stats,
                "frame_periods_ms": frame_periods_ms,
                "execution_times_ms": execution_times_ms,
            }

            return results

    def _do_real_work(self):
        """Perform actual computational work representing real processing."""
        # Simulate image processing or data analysis work

        # Matrix-like computations
        data = [0.0] * 1000
        for i in range(len(data)):
            data[i] = math.sin(i * 0.01) * math.cos(i * 0.02)

        self._work_result = sum(math.sqrt(abs(x) + 1.0) for x in data)


class LoadOperator(Operator):
    """
    Load operator that consumes CPU resources to create contention.
    """

    def __init__(self, fragment, load_duration_ms: float = 10.0, *args, **kwargs):
        self.load_duration_ms = load_duration_ms  # Duration in milliseconds
        self.iterations = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("load_data")

    def compute(self, op_input, op_output, context):
        # Perform CPU-intensive work
        self._consume_cpu()

        self.iterations += 1
        op_output.emit({"iteration": self.iterations}, "load_data")

    def _consume_cpu(self):
        """Consume CPU for the specified duration in milliseconds."""
        # Convert milliseconds to seconds
        work_duration = self.load_duration_ms / 1000.0
        end_time = time.perf_counter() + work_duration

        # Busy loop to consume CPU
        dummy = 0
        while time.perf_counter() < end_time:
            dummy += 1
            if dummy > 1000000:  # Prevent overflow
                dummy = 0


class DataSinkOperator(Operator):
    """
    Simple sink operator to receive data from other operators.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.received_count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("data", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        data = op_input.receive("data")
        if data is not None:
            self.received_count += 1


class RealtimeThreadBenchmark(Application):
    """
    Benchmark application to test real-time thread scheduling effectiveness.
    """

    def __init__(
        self,
        target_fps: int = 60,
        duration_seconds: int = 30,
        use_realtime: bool = False,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.SCHED_DEADLINE,
        load_duration_ms: float = 10.0,
        *args,
        **kwargs,
    ):
        self.target_fps = target_fps
        self.duration_seconds = duration_seconds
        self.use_realtime = use_realtime
        self.scheduling_policy = scheduling_policy
        self.load_duration_ms = load_duration_ms

        super().__init__(*args, **kwargs)

    def compose(self):
        # Create operators
        # For SCHED_DEADLINE, don't add PeriodicCondition since it provides its own periodicity
        if self.use_realtime and self.scheduling_policy == SchedulingPolicy.SCHED_DEADLINE:
            target_op = TargetOperator(
                self,
                self.target_fps,
                name="target_op",
            )
        else:
            # For other policies or non-realtime, use PeriodicCondition for periodicity
            target_op = TargetOperator(
                self,
                self.target_fps,
                PeriodicCondition(self, int(1e9 / self.target_fps)),  # Period in nanoseconds
                name="target_op",
            )

        load_op1 = LoadOperator(
            self,
            self.load_duration_ms,
            name="load_op1",
        )

        load_op2 = LoadOperator(
            self,
            self.load_duration_ms,
            name="load_op2",
        )

        sink_op = DataSinkOperator(self, name="sink_op")

        # Store reference to target operator for statistics
        self.target_operator = target_op

        # Create thread pools with limited worker threads to create contention
        if self.use_realtime:
            # Realtime pool for target operator
            realtime_pool = self.make_thread_pool("realtime_pool", 1)

            if self.scheduling_policy == SchedulingPolicy.SCHED_DEADLINE:
                period_ns = int(1e9 / self.target_fps)
                deadline_ns = int(period_ns * 0.95)  # 95% of period
                runtime_ns = int(period_ns * 0.10)  # 10% of period

                realtime_pool.add_realtime(
                    target_op,
                    self.scheduling_policy,
                    pin_operator=True,
                    pin_cores=[0],  # Pin to core 0
                    sched_runtime=runtime_ns,
                    sched_deadline=deadline_ns,
                    sched_period=period_ns,
                )
            elif self.scheduling_policy in [SchedulingPolicy.SCHED_FIFO, SchedulingPolicy.SCHED_RR]:
                realtime_pool.add_realtime(
                    target_op,
                    self.scheduling_policy,
                    pin_operator=True,
                    pin_cores=[0],  # Pin to core 0
                    sched_priority=99,  # High priority
                )

            # Regular pool for load operators (competing for resources)
            load_pool = self.make_thread_pool("load_pool", 1)
            load_pool.add(load_op1, pin_cores=[1])
            load_pool.add(load_op2, pin_cores=[1])
        else:
            # All operators share the same pool (no realtime scheduling)
            shared_pool = self.make_thread_pool("shared_pool", 2)
            shared_pool.add(target_op, pin_cores=[0, 1])
            shared_pool.add(load_op1, pin_cores=[0, 1])
            shared_pool.add(load_op2, pin_cores=[0, 1])

        # Connect operators
        self.add_flow(target_op, sink_op, {("frame", "data")})
        self.add_flow(load_op1, sink_op, {("load_data", "data")})
        self.add_flow(load_op2, sink_op, {("load_data", "data")})

    def get_benchmark_results(self):
        """Get benchmark results and raw data from the target operator."""
        return self.target_operator.get_statistics()


def run_benchmark(
    target_fps: int,
    duration_seconds: int,
    use_realtime: bool,
    scheduling_policy: SchedulingPolicy,
    load_duration_ms: float,
):
    """Run a single benchmark configuration."""

    print(f"\n{'=' * 60}")
    print("Running benchmark:")
    print(f"  Target FPS: {target_fps}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Realtime: {use_realtime}")
    print(f"  Policy: {scheduling_policy.name if use_realtime else 'Normal'}")
    print(f"  Load Duration: {load_duration_ms}ms per operator call")
    print("  EBS Worker Threads: 3")
    print(f"{'=' * 60}")

    app = RealtimeThreadBenchmark(
        target_fps=target_fps,
        duration_seconds=duration_seconds,
        use_realtime=use_realtime,
        scheduling_policy=scheduling_policy,
        load_duration_ms=load_duration_ms,
    )

    # Calculate scheduler timeout with reasonable buffer
    # Account for: startup time, shutdown time, and measurement overhead
    startup_buffer_ms = 1000  # 1s for application startup
    shutdown_buffer_ms = 1000  # 1s for graceful shutdown
    # Overhead scales with duration but caps at 5s to avoid excessive timeouts
    overhead_buffer_ms = min(max(500, duration_seconds * 20), 5000)  # 20ms/s, max 5s

    max_duration_ms = (
        duration_seconds * 1000 + startup_buffer_ms + shutdown_buffer_ms + overhead_buffer_ms
    )

    scheduler = EventBasedScheduler(
        app,
        worker_thread_number=3,  # Just enough to service our thread pools efficiently
        name="benchmark_scheduler",
        max_duration_ms=max_duration_ms,
    )
    app.scheduler(scheduler)

    # Run the application
    start_time = time.time()
    app.run()
    end_time = time.time()

    # Get results and raw data
    results = app.get_benchmark_results()

    results["actual_duration_s"] = end_time - start_time
    results["use_realtime"] = use_realtime
    results["scheduling_policy"] = scheduling_policy.name if use_realtime else "Normal"
    results["load_duration_ms"] = load_duration_ms

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    print("\nBenchmark Results:")
    rt_status = "RT" if results["use_realtime"] else "Normal"
    print(f"  Configuration: {results['scheduling_policy']} ({rt_status})")
    print(f"  Target FPS: {results['target_fps']:.1f}")

    # KEY METRICS FIRST - Frame timing consistency
    if results.get("frame_period_stats"):
        fps = results["frame_period_stats"]
        target_period_ms = 1000.0 / results["target_fps"]
        print(f"  ★ Frame Period Std Dev: {fps['std_ms']:.3f}ms  ← KEY METRIC")
        print(f"  Frame Period Mean: {fps['mean_ms']:.3f}ms (Target: {target_period_ms:.1f}ms)")

    # SECONDARY METRICS - Execution consistency
    if results.get("execution_time_stats"):
        ets = results["execution_time_stats"]
        print(f"  Execution Time Std Dev: {ets['std_ms']:.3f}ms")
        print(f"  Execution Time Mean: {ets['mean_ms']:.3f}ms")

    # DETAILED RANGES - Less critical but informative
    if results.get("frame_period_stats") and results.get("execution_time_stats"):
        fps = results["frame_period_stats"]
        ets = results["execution_time_stats"]
        print(f"  Frame Period Min/Max: {fps['min_ms']:.1f}ms / {fps['max_ms']:.1f}ms")
        print(f"  Execution Range: {ets['min_ms']:.3f}ms - {ets['max_ms']:.3f}ms")

    # TEST VALIDATION INFO - Basic metadata last
    print(f"  Frame Count: {results['frame_count']}")
    print(f"  Total Duration: {results['total_duration_s']:.2f}s")
    print(f"  Load Duration: {results['load_duration_ms']:.1f}ms per call")


def create_histogram_plots(
    normal_results, realtime_results, target_fps, output_dir="/tmp/benchmark_plots"
):
    """Create time series plots showing frame periods and execution times over time."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract raw data from results
    normal_periods = normal_results["frame_periods_ms"]
    normal_exec = normal_results["execution_times_ms"]
    realtime_periods = realtime_results["frame_periods_ms"]
    realtime_exec = realtime_results["execution_times_ms"]

    # Convert raw data to milliseconds
    target_period_ms = 1000.0 / target_fps  # Target period in ms

    # 1. Frame Period Over Time (Full + Zoomed + Execution)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

    # Calculate time arrays
    normal_period_time = np.arange(len(normal_periods)) * (target_period_ms / 1000)  # seconds
    realtime_period_time = np.arange(len(realtime_periods)) * (target_period_ms / 1000)  # seconds

    normal_exec_time = np.arange(len(normal_exec)) * (target_period_ms / 1000)  # seconds
    realtime_exec_time = np.arange(len(realtime_exec)) * (target_period_ms / 1000)  # seconds

    # Plot frame periods over time
    ax1.scatter(normal_period_time, normal_periods, alpha=0.6, color="red", s=2, label="Normal")
    ax1.scatter(
        realtime_period_time, realtime_periods, alpha=0.6, color="blue", s=2, label="SCHED_DEADLINE"
    )

    # Add target line
    ax1.axhline(
        y=target_period_ms,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_period_ms:.1f}ms)",
    )

    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Frame Period (ms)")
    ax1.set_title(f"Frame Period Over Time - Full Range (Target: {target_fps} FPS)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot frame periods over time - ZOOMED VIEW (ax2)
    ax2.scatter(normal_period_time, normal_periods, alpha=0.6, color="red", s=2, label="Normal")
    ax2.scatter(
        realtime_period_time, realtime_periods, alpha=0.6, color="blue", s=2, label="SCHED_DEADLINE"
    )

    # Add target line for zoomed view
    ax2.axhline(
        y=target_period_ms,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_period_ms:.1f}ms)",
    )

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frame Period (ms)")
    ax2.set_title(f"Frame Period Over Time - Zoomed (Target: {target_fps} FPS)")
    ax2.set_ylim(target_period_ms * 0.95, target_period_ms * 1.05)  # Zoom to ±5% of target
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot execution times over time (ax3)
    ax3.scatter(normal_exec_time, normal_exec, alpha=0.6, color="red", s=2, label="Normal")
    ax3.scatter(
        realtime_exec_time, realtime_exec, alpha=0.6, color="blue", s=2, label="SCHED_DEADLINE"
    )

    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Execution Time (ms)")
    ax3.set_title("Execution Time Over Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timing_over_time.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Simple Histograms
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Frame period histogram
    bins_period = np.linspace(
        min(min(normal_periods), min(realtime_periods)) * 0.9,
        max(max(normal_periods), max(realtime_periods)) * 1.1,
        150,  # Finer resolution
    )

    ax1.hist(
        normal_periods,
        bins=bins_period,
        alpha=0.6,
        color="red",
        label="Normal",
        density=False,
    )
    ax1.hist(
        realtime_periods,
        bins=bins_period,
        alpha=0.6,
        color="blue",
        label="SCHED_DEADLINE",
        density=False,
    )
    ax1.axvline(
        x=target_period_ms,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_period_ms:.1f}ms)",
    )

    ax1.set_xlabel("Frame Period (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title("Frame Period Distribution - Full Range")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Frame period histogram - ZOOMED (ax2) with finer bins
    bins_period_zoomed = np.linspace(
        target_period_ms * 0.95, target_period_ms * 1.05, 300
    )  # ±5% of target with fine resolution

    ax2.hist(
        normal_periods,
        bins=bins_period_zoomed,
        alpha=0.6,
        color="red",
        label="Normal",
        density=False,
    )
    ax2.hist(
        realtime_periods,
        bins=bins_period_zoomed,
        alpha=0.6,
        color="blue",
        label="SCHED_DEADLINE",
        density=False,
    )
    ax2.axvline(
        x=target_period_ms,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Target ({target_period_ms:.1f}ms)",
    )

    ax2.set_xlabel("Frame Period (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("Frame Period Distribution - Zoomed")
    ax2.set_xlim(target_period_ms * 0.95, target_period_ms * 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Execution time histogram (ax3)
    bins_exec = np.linspace(
        min(min(normal_exec), min(realtime_exec)) * 0.9,
        max(max(normal_exec), max(realtime_exec)) * 1.1,
        150,  # Finer resolution
    )

    ax3.hist(
        normal_exec,
        bins=bins_exec,
        alpha=0.6,
        color="red",
        label="Normal",
        density=False,
    )
    ax3.hist(
        realtime_exec,
        bins=bins_exec,
        alpha=0.6,
        color="blue",
        label="SCHED_DEADLINE",
        density=False,
    )

    ax3.set_xlabel("Execution Time (ms)")
    ax3.set_ylabel("Count")
    ax3.set_title("Execution Time Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "simple_histograms.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nTiming plots saved to: {output_dir}")
    print("Generated plots:")
    print("  - timing_over_time.png (raw data points over time)")
    print("  - simple_histograms.png (distribution without overlays)")


def main():
    parser = argparse.ArgumentParser(description="Holoscan Real-time Thread Benchmark")
    parser.add_argument(
        "--target-fps",
        type=int,
        choices=[30, 60],
        default=60,
        help="Target FPS for the benchmark (30 or 60)",
    )
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration in seconds")
    parser.add_argument(
        "--scheduling-policy",
        choices=["SCHED_DEADLINE", "SCHED_FIFO", "SCHED_RR"],
        default="SCHED_DEADLINE",
        help="Real-time scheduling policy to compare against normal scheduling",
    )
    parser.add_argument(
        "--load-duration-ms",
        type=float,
        default=20.0,
        help="CPU work duration per load operator call (milliseconds)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="/tmp/benchmark_plots",
        help="Directory to save benchmark plots (default: /tmp/benchmark_plots)",
    )

    args = parser.parse_args()

    policy_map = {
        "SCHED_DEADLINE": SchedulingPolicy.SCHED_DEADLINE,
        "SCHED_FIFO": SchedulingPolicy.SCHED_FIFO,
        "SCHED_RR": SchedulingPolicy.SCHED_RR,
    }
    scheduling_policy = policy_map[args.scheduling_policy]

    print("Running real-time thread scheduling comparison...")

    # Run without real-time scheduling
    normal_results = run_benchmark(
        args.target_fps,
        args.duration,
        False,
        scheduling_policy,
        args.load_duration_ms,
    )

    # Run with real-time scheduling
    realtime_results = run_benchmark(
        args.target_fps,
        args.duration,
        True,
        scheduling_policy,
        args.load_duration_ms,
    )

    # Create histogram plots
    create_histogram_plots(
        normal_results,
        realtime_results,
        args.target_fps,
        args.plot_dir,
    )

    print_results(normal_results)
    print_results(realtime_results)

    print(f"\n{'=' * 65}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 65}")
    print("                        Normal    Real-time    Improvement")
    print("-" * 65)

    if normal_results.get("frame_period_stats") and realtime_results.get("frame_period_stats"):
        normal_period_std = normal_results["frame_period_stats"]["std_ms"]
        realtime_period_std = realtime_results["frame_period_stats"]["std_ms"]
        period_std_improvement = (realtime_period_std / normal_period_std - 1) * 100
        print(
            f"★ Frame Period Std Dev: {normal_period_std:7.3f}    "
            f"{realtime_period_std:9.3f}    {period_std_improvement:+7.1f}% ★"
        )

    # Secondary metrics
    if normal_results.get("execution_time_stats") and realtime_results.get("execution_time_stats"):
        normal_exec_std = normal_results["execution_time_stats"]["std_ms"]
        realtime_exec_std = realtime_results["execution_time_stats"]["std_ms"]
        exec_std_improvement = (realtime_exec_std / normal_exec_std - 1) * 100
        print(
            f"  Exec Time Std Dev:     {normal_exec_std:7.3f}    "
            f"{realtime_exec_std:9.3f}    {exec_std_improvement:+7.1f}%"
        )


if __name__ == "__main__":
    main()
