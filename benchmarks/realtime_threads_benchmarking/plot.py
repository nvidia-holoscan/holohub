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
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for containers


def load_benchmark_data(json_file):
    """Load benchmark data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def create_histogram_plots(normal_results, realtime_results, target_fps, output_dir="/tmp/benchmark_plots"):
    """Create time series plots showing frame periods and execution times over time."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract raw data from results
    normal_periods = normal_results.get("frame_periods_ms", [])
    normal_exec = normal_results.get("execution_times_ms", [])
    realtime_periods = realtime_results.get("frame_periods_ms", [])
    realtime_exec = realtime_results.get("execution_times_ms", [])

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
    if normal_periods and realtime_periods:
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
    if normal_exec and realtime_exec:
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


def print_results(results):
    """Print benchmark results in a formatted way."""
    print("\nBenchmark Results:")
    rt_status = "RT" if results.get("use_realtime", False) else "Normal"
    print(f"  Configuration: {results.get('scheduling_policy', 'Unknown')} ({rt_status})")
    print(f"  Target FPS: {results.get('target_fps', 0):.1f}")

    # KEY METRICS FIRST - Frame timing consistency
    frame_period_stats = results.get("frame_period_stats", {})
    if frame_period_stats:
        fps = frame_period_stats
        target_period_ms = 1000.0 / results.get("target_fps", 60)
        print(f"  ★ Frame Period Std Dev: {fps.get('std_ms', 0):.3f}ms  ← KEY METRIC")
        print(f"  Frame Period Mean: {fps.get('mean_ms', 0):.3f}ms (Target: {target_period_ms:.1f}ms)")

    # SECONDARY METRICS - Execution consistency
    execution_time_stats = results.get("execution_time_stats", {})
    if execution_time_stats:
        ets = execution_time_stats
        print(f"  Execution Time Std Dev: {ets.get('std_ms', 0):.3f}ms")
        print(f"  Execution Time Mean: {ets.get('mean_ms', 0):.3f}ms")

    # DETAILED RANGES - Less critical but informative
    if frame_period_stats and execution_time_stats:
        fps = frame_period_stats
        ets = execution_time_stats
        print(f"  Frame Period Min/Max: {fps.get('min_ms', 0):.1f}ms / {fps.get('max_ms', 0):.1f}ms")
        print(f"  Execution Range: {ets.get('min_ms', 0):.3f}ms - {ets.get('max_ms', 0):.3f}ms")

    # TEST VALIDATION INFO - Basic metadata last
    print(f"  Frame Count: {results.get('frame_count', 0)}")
    print(f"  Total Duration: {results.get('total_duration_s', 0):.2f}s")
    print(f"  Load Duration: {results.get('load_duration_ms', 0):.1f}ms per call")


def main():
    parser = argparse.ArgumentParser(description="Plot real-time thread benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with benchmark results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/benchmark_plots",
        help="Directory to save benchmark plots (default: /tmp/benchmark_plots)"
    )

    args = parser.parse_args()

    # Load benchmark data
    data = load_benchmark_data(args.input)
    
    normal_results = data.get("normal", {})
    realtime_results = data.get("realtime", {})
    
    if not normal_results or not realtime_results:
        print("Error: JSON file must contain both 'normal' and 'realtime' sections")
        return 1

    target_fps = normal_results.get("target_fps", 60)

    # Create histogram plots
    create_histogram_plots(
        normal_results,
        realtime_results,
        target_fps,
        args.output_dir,
    )

    print_results(normal_results)
    print_results(realtime_results)

    print(f"\n{'=' * 65}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 65}")
    print("                        Normal    Real-time    Improvement")
    print("-" * 65)

    normal_frame_period_stats = normal_results.get("frame_period_stats", {})
    realtime_frame_period_stats = realtime_results.get("frame_period_stats", {})
    
    if normal_frame_period_stats and realtime_frame_period_stats:
        normal_period_std = normal_frame_period_stats.get("std_ms", 0)
        realtime_period_std = realtime_frame_period_stats.get("std_ms", 0)
        if normal_period_std > 0:
            period_std_improvement = (realtime_period_std / normal_period_std - 1) * 100
            print(
                f"★ Frame Period Std Dev: {normal_period_std:7.3f}    "
                f"{realtime_period_std:9.3f}    {period_std_improvement:+7.1f}% ★"
            )

    # Secondary metrics
    normal_execution_time_stats = normal_results.get("execution_time_stats", {})
    realtime_execution_time_stats = realtime_results.get("execution_time_stats", {})
    
    if normal_execution_time_stats and realtime_execution_time_stats:
        normal_exec_std = normal_execution_time_stats.get("std_ms", 0)
        realtime_exec_std = realtime_execution_time_stats.get("std_ms", 0)
        if normal_exec_std > 0:
            exec_std_improvement = (realtime_exec_std / normal_exec_std - 1) * 100
            print(
                f"  Exec Time Std Dev:     {normal_exec_std:7.3f}    "
                f"{realtime_exec_std:9.3f}    {exec_std_improvement:+7.1f}%"
            )

    return 0


if __name__ == "__main__":
    exit(main())
