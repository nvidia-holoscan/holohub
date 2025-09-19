#!/usr/bin/env python3
"""
Plotting script for C++ realtime thread benchmark results.
Reads JSON output from C++ benchmark and generates the same plots as Python version.
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for containers


def create_histogram_plots(json_data, output_dir="/tmp/benchmark_plots"):
    """Create time series plots showing frame periods and execution times over time."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract configuration
    config = json_data["benchmark_config"]
    target_fps = config["target_fps"]
    target_period_ms = 1000.0 / target_fps

    # Extract raw data from JSON
    period_stats = json_data["period_statistics"]
    normal_periods = period_stats["non_realtime"]["raw_data"]
    realtime_periods = period_stats["realtime"]["raw_data"]

    exec_stats = json_data["execution_time_statistics"]
    normal_exec = exec_stats["non_realtime"]["raw_data"]
    realtime_exec = exec_stats["realtime"]["raw_data"]

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
    parser = argparse.ArgumentParser(description="Generate plots from C++ benchmark JSON results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file from C++ benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/benchmark_plots",
        help="Output directory for plots (default: /tmp/benchmark_plots)"
    )

    args = parser.parse_args()

    # Read JSON data
    try:
        with open(args.input, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find input file {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON file {args.input}: {e}")
        sys.exit(1)

    # Generate plots
    create_histogram_plots(json_data, args.output_dir)


if __name__ == "__main__":
    main()
