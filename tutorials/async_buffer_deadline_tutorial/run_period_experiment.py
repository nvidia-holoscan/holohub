#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
NUM_MESSAGES = 1500
DISCARD_FIRST = 100
DISCARD_LAST = 100
PERIODS = list(range(20, 101, 10))  # 20, 30, 40, 50, 60, 70, 80, 90, 100
RESULTS_DIR = "period_variation"


def run_experiment(tx1_period, tx2_period, enable_async_buffer, mechanism_name):
    """Run a single experiment with given parameters"""
    print(
        f"Running experiment: tx1_period={tx1_period}ms, tx2_period={tx2_period}ms, mechanism={mechanism_name}"
    )

    # Prepare command
    if enable_async_buffer:
        run_args = f"-m {NUM_MESSAGES} -a -x {tx1_period} -y {tx2_period}"
    else:
        run_args = f"-m {NUM_MESSAGES} -x {tx1_period} -y {tx2_period}"

    cmd = [
        "../../holohub",
        "run",
        "async_buffer_deadline",
        "--run-args",
        run_args,
        "--as-root",
        "--docker-opts",
        "--ulimit rtprio=99 --cap-add=CAP_SYS_NICE",
    ]

    # Run the experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Experiment failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Experiment timed out")
        return False

    # Copy CSV files to results directory
    mechanism_dir = os.path.join(RESULTS_DIR, mechanism_name)
    os.makedirs(mechanism_dir, exist_ok=True)

    # Generate unique filename based on periods
    suffix = f"_tx1_{tx1_period}_tx2_{tx2_period}.csv"

    csv_files = ["tx1.csv", "tx2.csv", "rx_in1_periods.csv", "rx_in2_periods.csv"]
    for csv_file in csv_files:
        src = os.path.join(
            "../../build/async_buffer_deadline/applications/async_buffer_deadline/", csv_file
        )
        dst = os.path.join(mechanism_dir, csv_file.replace(".csv", suffix))
        if os.path.exists(src):
            shutil.copy2(src, dst)

    return True


def analyze_latency_data(csv_file):
    """Analyze latency data and return max latency"""
    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)

    # Discard first and last messages
    if len(df) > DISCARD_FIRST + DISCARD_LAST:
        df = df.iloc[DISCARD_FIRST:-DISCARD_LAST]

    if len(df) == 0:
        return None

    return df["latency_ms"].max()


def analyze_period_data(csv_file):
    """Analyze period data and return max period"""
    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)

    # Discard first and last messages
    if len(df) > DISCARD_FIRST + DISCARD_LAST:
        df = df.iloc[DISCARD_FIRST:-DISCARD_LAST]

    if len(df) == 0:
        return None

    return df["period_ms"].max()


def run_all_experiments():
    """Run all experiments for both scenarios"""

    # Scenario 1: Keep tx1 period constant at 20ms, vary tx2 period
    print("=== Scenario 1: Fixed tx1 period (20ms), varying tx2 period ===")
    tx1_fixed_results = {
        "default_buffer": {
            "tx1_max_latency": [],
            "tx2_max_latency": [],
            "in1_max_period": [],
            "in2_max_period": [],
        },
        "async_lockfree_buffer": {
            "tx1_max_latency": [],
            "tx2_max_latency": [],
            "in1_max_period": [],
            "in2_max_period": [],
        },
    }

    for tx2_period in PERIODS:
        print(f"\n--- Testing tx2_period = {tx2_period}ms ---")

        # Test default buffer
        success = run_experiment(20, tx2_period, False, "default_buffer")
        if success:
            # Analyze tx1 latency (should be affected by tx2 period)
            tx1_csv = f"period_variation/default_buffer/tx1_tx1_20_tx2_{tx2_period}.csv"
            max_latency = analyze_latency_data(tx1_csv)
            tx1_fixed_results["default_buffer"]["tx1_max_latency"].append(max_latency)

            # Analyze tx2 latency
            tx2_csv = f"period_variation/default_buffer/tx2_tx1_20_tx2_{tx2_period}.csv"
            max_latency = analyze_latency_data(tx2_csv)
            tx1_fixed_results["default_buffer"]["tx2_max_latency"].append(max_latency)

            # Analyze in1 periods
            in1_csv = f"period_variation/default_buffer/rx_in1_periods_tx1_20_tx2_{tx2_period}.csv"
            max_period = analyze_period_data(in1_csv)
            tx1_fixed_results["default_buffer"]["in1_max_period"].append(max_period)

            # Analyze in2 periods
            in2_csv = f"period_variation/default_buffer/rx_in2_periods_tx1_20_tx2_{tx2_period}.csv"
            max_period = analyze_period_data(in2_csv)
            tx1_fixed_results["default_buffer"]["in2_max_period"].append(max_period)

        # Test async lock-free buffer
        success = run_experiment(20, tx2_period, True, "async_lockfree_buffer")
        if success:
            # Analyze tx1 latency
            tx1_csv = f"period_variation/async_lockfree_buffer/tx1_tx1_20_tx2_{tx2_period}.csv"
            max_latency = analyze_latency_data(tx1_csv)
            tx1_fixed_results["async_lockfree_buffer"]["tx1_max_latency"].append(max_latency)

            # Analyze tx2 latency
            tx2_csv = f"period_variation/async_lockfree_buffer/tx2_tx1_20_tx2_{tx2_period}.csv"
            max_latency = analyze_latency_data(tx2_csv)
            tx1_fixed_results["async_lockfree_buffer"]["tx2_max_latency"].append(max_latency)

            # Analyze in1 periods
            in1_csv = (
                f"period_variation/async_lockfree_buffer/rx_in1_periods_tx1_20_tx2_{tx2_period}.csv"
            )
            max_period = analyze_period_data(in1_csv)
            tx1_fixed_results["async_lockfree_buffer"]["in1_max_period"].append(max_period)

            # Analyze in2 periods
            in2_csv = (
                f"period_variation/async_lockfree_buffer/rx_in2_periods_tx1_20_tx2_{tx2_period}.csv"
            )
            max_period = analyze_period_data(in2_csv)
            tx1_fixed_results["async_lockfree_buffer"]["in2_max_period"].append(max_period)

    # Scenario 2: Keep tx2 period constant at 20ms, vary tx1 period
    print("\n=== Scenario 2: Fixed tx2 period (20ms), varying tx1 period ===")
    tx2_fixed_results = {
        "default_buffer": {
            "tx1_max_latency": [],
            "tx2_max_latency": [],
            "in1_max_period": [],
            "in2_max_period": [],
        },
        "async_lockfree_buffer": {
            "tx1_max_latency": [],
            "tx2_max_latency": [],
            "in1_max_period": [],
            "in2_max_period": [],
        },
    }

    for tx1_period in PERIODS:
        print(f"\n--- Testing tx1_period = {tx1_period}ms ---")

        # Test default buffer
        success = run_experiment(tx1_period, 20, False, "default_buffer")
        if success:
            # Analyze tx1 latency
            tx1_csv = f"period_variation/default_buffer/tx1_tx1_{tx1_period}_tx2_20.csv"
            max_latency = analyze_latency_data(tx1_csv)
            tx2_fixed_results["default_buffer"]["tx1_max_latency"].append(max_latency)

            # Analyze tx2 latency (should be affected by tx1 period)
            tx2_csv = f"period_variation/default_buffer/tx2_tx1_{tx1_period}_tx2_20.csv"
            max_latency = analyze_latency_data(tx2_csv)
            tx2_fixed_results["default_buffer"]["tx2_max_latency"].append(max_latency)

            # Analyze in1 periods
            in1_csv = f"period_variation/default_buffer/rx_in1_periods_tx1_{tx1_period}_tx2_20.csv"
            max_period = analyze_period_data(in1_csv)
            tx2_fixed_results["default_buffer"]["in1_max_period"].append(max_period)

            # Analyze in2 periods
            in2_csv = f"period_variation/default_buffer/rx_in2_periods_tx1_{tx1_period}_tx2_20.csv"
            max_period = analyze_period_data(in2_csv)
            tx2_fixed_results["default_buffer"]["in2_max_period"].append(max_period)

        # Test async lock-free buffer
        success = run_experiment(tx1_period, 20, True, "async_lockfree_buffer")
        if success:
            # Analyze tx1 latency
            tx1_csv = f"period_variation/async_lockfree_buffer/tx1_tx1_{tx1_period}_tx2_20.csv"
            max_latency = analyze_latency_data(tx1_csv)
            tx2_fixed_results["async_lockfree_buffer"]["tx1_max_latency"].append(max_latency)

            # Analyze tx2 latency
            tx2_csv = f"period_variation/async_lockfree_buffer/tx2_tx1_{tx1_period}_tx2_20.csv"
            max_latency = analyze_latency_data(tx2_csv)
            tx2_fixed_results["async_lockfree_buffer"]["tx2_max_latency"].append(max_latency)

            # Analyze in1 periods
            in1_csv = (
                f"period_variation/async_lockfree_buffer/rx_in1_periods_tx1_{tx1_period}_tx2_20.csv"
            )
            max_period = analyze_period_data(in1_csv)
            tx2_fixed_results["async_lockfree_buffer"]["in1_max_period"].append(max_period)

            # Analyze in2 periods
            in2_csv = (
                f"period_variation/async_lockfree_buffer/rx_in2_periods_tx1_{tx1_period}_tx2_20.csv"
            )
            max_period = analyze_period_data(in2_csv)
            tx2_fixed_results["async_lockfree_buffer"]["in2_max_period"].append(max_period)

    return tx1_fixed_results, tx2_fixed_results


def create_plots(tx1_fixed_results, tx2_fixed_results, font_size=12, title_font_size=14):
    """Create all the plots with customizable parameters"""

    # Fixed plot parameters
    fig_width = 12
    fig_height = 8
    dpi = 300

    # Set up the plotting style with custom font sizes
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": title_font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size + 1,
        }
    )

    # Plot 1: TX1 Max Latency (Fixed TX1 period, varying TX2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    x = np.arange(len(PERIODS))
    width = 0.35

    db_tx1_latency = tx1_fixed_results["default_buffer"]["tx1_max_latency"]
    alfb_tx1_latency = tx1_fixed_results["async_lockfree_buffer"]["tx1_max_latency"]

    ax.bar(x - width / 2, db_tx1_latency, width, label="Default Buffer", alpha=0.8)
    ax.bar(x + width / 2, alfb_tx1_latency, width, label="Async Lock-free Buffer", alpha=0.8)

    ax.set_xlabel("TX2 Period (ms)", fontsize=font_size)
    ax.set_ylabel("Max Latency (ms)", fontsize=font_size)
    ax.set_title(
        "TX1 Message Max Latency vs TX2 Period (Fixed TX1 Period = 20ms)\n Fixed RX Period = 10ms",
        fontsize=title_font_size,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/tx1_latency_vs_tx2_period.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # Plot 2: TX2 Max Latency (Fixed TX2 period, varying TX1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    db_tx2_latency = tx2_fixed_results["default_buffer"]["tx2_max_latency"]
    alfb_tx2_latency = tx2_fixed_results["async_lockfree_buffer"]["tx2_max_latency"]

    ax.bar(x - width / 2, db_tx2_latency, width, label="Default Buffer", alpha=0.8)
    ax.bar(x + width / 2, alfb_tx2_latency, width, label="Async Lock-free Buffer", alpha=0.8)

    ax.set_xlabel("TX1 Period (ms)", fontsize=font_size)
    ax.set_ylabel("Max Latency (ms)", fontsize=font_size)
    ax.set_title(
        "TX2 Message Max Latency vs TX1 Period (Fixed TX2 Period = 20ms)\n Fixed RX Period = 10ms",
        fontsize=title_font_size,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/tx2_latency_vs_tx1_period.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # Plot 3: IN1 Max Period (Fixed TX1 period, varying TX2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    db_in1_period = tx1_fixed_results["default_buffer"]["in1_max_period"]
    alfb_in1_period = tx1_fixed_results["async_lockfree_buffer"]["in1_max_period"]

    ax.bar(x - width / 2, db_in1_period, width, label="Default Buffer", alpha=0.8)
    ax.bar(x + width / 2, alfb_in1_period, width, label="Async Lock-free Buffer", alpha=0.8)

    ax.set_xlabel("TX2 Period (ms)", fontsize=font_size)
    ax.set_ylabel("Max Message Interval (ms)", fontsize=font_size)
    ax.set_title(
        "IN1 (from TX1) Max Message Interval at RX vs TX2 Period (Fixed TX1 Period = 20ms)\n Fixed RX Period = 10ms",
        fontsize=title_font_size,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/in1_period_vs_tx2_period.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # Plot 4: IN2 Max Period (Fixed TX2 period, varying TX1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    db_in2_period = tx2_fixed_results["default_buffer"]["in2_max_period"]
    alfb_in2_period = tx2_fixed_results["async_lockfree_buffer"]["in2_max_period"]

    ax.bar(x - width / 2, db_in2_period, width, label="Default Buffer", alpha=0.8)
    ax.bar(x + width / 2, alfb_in2_period, width, label="Async Lock-free Buffer", alpha=0.8)

    ax.set_xlabel("TX1 Period (ms)", fontsize=font_size)
    ax.set_ylabel("Max Message Interval (ms)", fontsize=font_size)
    ax.set_title(
        "IN2 (from TX2) Max Message Interval at RX vs TX1 Period (Fixed TX2 Period = 20ms)\n Fixed RX Period = 10ms",
        fontsize=title_font_size,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/in2_period_vs_tx1_period.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def collect_data_from_csv():
    """Collect data from existing CSV files"""

    # Scenario 1: Fixed TX1 period (20ms), varying TX2 period
    tx1_fixed_results = {
        "default_buffer": {
            "tx1_max_latency": [],
            "tx2_max_latency": [],
            "in1_max_period": [],
            "in2_max_period": [],
        },
        "async_lockfree_buffer": {
            "tx1_max_latency": [],
            "tx2_max_latency": [],
            "in1_max_period": [],
            "in2_max_period": [],
        },
    }

    for tx2_period in PERIODS:
        # Default buffer
        tx1_csv = f"{RESULTS_DIR}/default_buffer/tx1_tx1_20_tx2_{tx2_period}.csv"
        max_latency = analyze_latency_data(tx1_csv)
        tx1_fixed_results["default_buffer"]["tx1_max_latency"].append(max_latency)

        tx2_csv = f"{RESULTS_DIR}/default_buffer/tx2_tx1_20_tx2_{tx2_period}.csv"
        max_latency = analyze_latency_data(tx2_csv)
        tx1_fixed_results["default_buffer"]["tx2_max_latency"].append(max_latency)

        in1_csv = f"{RESULTS_DIR}/default_buffer/rx_in1_periods_tx1_20_tx2_{tx2_period}.csv"
        max_period = analyze_period_data(in1_csv)
        tx1_fixed_results["default_buffer"]["in1_max_period"].append(max_period)

        in2_csv = f"{RESULTS_DIR}/default_buffer/rx_in2_periods_tx1_20_tx2_{tx2_period}.csv"
        max_period = analyze_period_data(in2_csv)
        tx1_fixed_results["default_buffer"]["in2_max_period"].append(max_period)

        # Async lock-free buffer
        tx1_csv = f"{RESULTS_DIR}/async_lockfree_buffer/tx1_tx1_20_tx2_{tx2_period}.csv"
        max_latency = analyze_latency_data(tx1_csv)
        tx1_fixed_results["async_lockfree_buffer"]["tx1_max_latency"].append(max_latency)

        tx2_csv = f"{RESULTS_DIR}/async_lockfree_buffer/tx2_tx1_20_tx2_{tx2_period}.csv"
        max_latency = analyze_latency_data(tx2_csv)
        tx1_fixed_results["async_lockfree_buffer"]["tx2_max_latency"].append(max_latency)

        in1_csv = f"{RESULTS_DIR}/async_lockfree_buffer/rx_in1_periods_tx1_20_tx2_{tx2_period}.csv"
        max_period = analyze_period_data(in1_csv)
        tx1_fixed_results["async_lockfree_buffer"]["in1_max_period"].append(max_period)

        in2_csv = f"{RESULTS_DIR}/async_lockfree_buffer/rx_in2_periods_tx1_20_tx2_{tx2_period}.csv"
        max_period = analyze_period_data(in2_csv)
        tx1_fixed_results["async_lockfree_buffer"]["in2_max_period"].append(max_period)

    # Scenario 2: Fixed TX2 period (20ms), varying TX1 period
    tx2_fixed_results = {
        "default_buffer": {
            "tx1_max_latency": [],
            "tx2_max_latency": [],
            "in1_max_period": [],
            "in2_max_period": [],
        },
        "async_lockfree_buffer": {
            "tx1_max_latency": [],
            "tx2_max_latency": [],
            "in1_max_period": [],
            "in2_max_period": [],
        },
    }

    for tx1_period in PERIODS:
        # Default buffer
        tx1_csv = f"{RESULTS_DIR}/default_buffer/tx1_tx1_{tx1_period}_tx2_20.csv"
        max_latency = analyze_latency_data(tx1_csv)
        tx2_fixed_results["default_buffer"]["tx1_max_latency"].append(max_latency)

        tx2_csv = f"{RESULTS_DIR}/default_buffer/tx2_tx1_{tx1_period}_tx2_20.csv"
        max_latency = analyze_latency_data(tx2_csv)
        tx2_fixed_results["default_buffer"]["tx2_max_latency"].append(max_latency)

        in1_csv = f"{RESULTS_DIR}/default_buffer/rx_in1_periods_tx1_{tx1_period}_tx2_20.csv"
        max_period = analyze_period_data(in1_csv)
        tx2_fixed_results["default_buffer"]["in1_max_period"].append(max_period)

        in2_csv = f"{RESULTS_DIR}/default_buffer/rx_in2_periods_tx1_{tx1_period}_tx2_20.csv"
        max_period = analyze_period_data(in2_csv)
        tx2_fixed_results["default_buffer"]["in2_max_period"].append(max_period)

        # Async lock-free buffer
        tx1_csv = f"{RESULTS_DIR}/async_lockfree_buffer/tx1_tx1_{tx1_period}_tx2_20.csv"
        max_latency = analyze_latency_data(tx1_csv)
        tx2_fixed_results["async_lockfree_buffer"]["tx1_max_latency"].append(max_latency)

        tx2_csv = f"{RESULTS_DIR}/async_lockfree_buffer/tx2_tx1_{tx1_period}_tx2_20.csv"
        max_latency = analyze_latency_data(tx2_csv)
        tx2_fixed_results["async_lockfree_buffer"]["tx2_max_latency"].append(max_latency)

        in1_csv = f"{RESULTS_DIR}/async_lockfree_buffer/rx_in1_periods_tx1_{tx1_period}_tx2_20.csv"
        max_period = analyze_period_data(in1_csv)
        tx2_fixed_results["async_lockfree_buffer"]["in1_max_period"].append(max_period)

        in2_csv = f"{RESULTS_DIR}/async_lockfree_buffer/rx_in2_periods_tx1_{tx1_period}_tx2_20.csv"
        max_period = analyze_period_data(in2_csv)
        tx2_fixed_results["async_lockfree_buffer"]["in2_max_period"].append(max_period)

    return tx1_fixed_results, tx2_fixed_results


def main():
    """Main function to run the experiment or generate plots"""
    parser = argparse.ArgumentParser(description="Period Variation Experiment")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate plots from existing CSV data without running experiments",
    )
    parser.add_argument(
        "--font-size", type=int, default=16, help="Font size for plot labels (default: 16)"
    )
    parser.add_argument(
        "--title-font-size", type=int, default=18, help="Font size for plot titles (default: 18)"
    )

    args = parser.parse_args()

    if args.plots_only:
        # Generate plots from existing data
        print("Generating plots from existing CSV data...")

        # Check if results directory exists
        if not os.path.exists(RESULTS_DIR):
            print(f"Error: {RESULTS_DIR} directory not found!")
            print(
                "Please run the experiment first or ensure CSV files are in the correct location."
            )
            return

        # Collect data from existing CSV files
        print("Collecting data from CSV files...")
        tx1_fixed_results, tx2_fixed_results = collect_data_from_csv()

        # Check if we have data
        if not any(tx1_fixed_results["default_buffer"]["tx1_max_latency"]):
            print("No data found! Please ensure CSV files exist in the period_variation directory.")
            return

        # Create plots with custom settings
        print("Creating plots...")
        create_plots(
            tx1_fixed_results,
            tx2_fixed_results,
            font_size=args.font_size,
            title_font_size=args.title_font_size,
        )

        print(f"\nPlots generated successfully in {RESULTS_DIR}/")
        print("Generated plots:")
        print("- tx1_latency_vs_tx2_period.png")
        print("- tx2_latency_vs_tx1_period.png")
        print("- in1_period_vs_tx2_period.png")
        print("- in2_period_vs_tx1_period.png")

    else:
        # Run the full experiment (default behavior)
        print("Starting period variation experiment...")
        print(f"Number of messages: {NUM_MESSAGES}")
        print(f"Discarding first {DISCARD_FIRST} and last {DISCARD_LAST} messages")
        print(f"Testing periods: {PERIODS}")

        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, "default_buffer"), exist_ok=True)
        os.makedirs(os.path.join(RESULTS_DIR, "async_lockfree_buffer"), exist_ok=True)

        # Run experiments
        tx1_fixed_results, tx2_fixed_results = run_all_experiments()

        # Create plots
        print("\nCreating plots...")
        create_plots(
            tx1_fixed_results,
            tx2_fixed_results,
            font_size=args.font_size,
            title_font_size=args.title_font_size,
        )

        print(f"\nExperiment completed! Results saved in {RESULTS_DIR}/")
        print("Generated plots:")
        print("- tx1_latency_vs_tx2_period.png")
        print("- tx2_latency_vs_tx1_period.png")
        print("- in1_period_vs_tx2_period.png")
        print("- in2_period_vs_tx1_period.png")


if __name__ == "__main__":
    main()
