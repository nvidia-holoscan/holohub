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
import re
import sys
from statistics import mean


def analyze_text_log(filename, discard=0):
    print(f"Analyzing Text log: {filename}")
    if discard > 0:
        print(f"Discarding first and last {discard} samples for stability.")

    # Format: (op_name, start_us, end_us) -> (next_op, start_us, end_us)
    # Regex to capture: (name,start,end)
    # We can calculate execution time = end - start

    latencies = {}  # op_name -> list of durations (ms)
    e2e_latencies = []  # list of (start_first_op, end_last_op) durations (ms)

    op_pattern = re.compile(r"\(([^,]+),(\d+),(\d+)\)")

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.endswith(":"):
                    continue

                matches = op_pattern.findall(line)
                if not matches:
                    continue

                # Calculate individual op latencies
                for name, start, end in matches:
                    # Timestamps are microseconds (16 digits, e.g. 1.76e15)
                    # Duration in ms = (end - start) / 1000.0
                    duration_ms = (int(end) - int(start)) / 1000.0
                    if name not in latencies:
                        latencies[name] = []
                    latencies[name].append(duration_ms)

                # End-to-end for this path
                if len(matches) >= 2:
                    first_start = int(matches[0][1])
                    last_end = int(matches[-1][2])
                    e2e_ms = (last_end - first_start) / 1000.0
                    e2e_latencies.append(e2e_ms)

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not latencies:
        print("No valid trace data found.")
        return

    print("\n--- Performance Report ---")
    print(
        f"{'Operator':<25} | {'Avg (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Count':<5}"
    )
    print("-" * 75)

    for op, durations in latencies.items():
        if discard > 0 and len(durations) > 2 * discard:
            durations = durations[discard:-discard]

        avg_d = mean(durations)
        min_d = min(durations)
        max_d = max(durations)
        count = len(durations)
        print(f"{op:<25} | {avg_d:<10.3f} | {min_d:<10.3f} | {max_d:<10.3f} | {count:<5}")

    if e2e_latencies:
        if discard > 0 and len(e2e_latencies) > 2 * discard:
            e2e_latencies = e2e_latencies[discard:-discard]

        print("-" * 75)
        print(
            f"{'End-to-End':<25} | {mean(e2e_latencies):<10.3f} | {min(e2e_latencies):<10.3f} | {max(e2e_latencies):<10.3f} | {len(e2e_latencies):<5}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze Holoscan Tracker Log")
    parser.add_argument("logfile", help="Path to the log file")
    parser.add_argument(
        "--discard", type=int, default=0, help="Number of samples to discard from start and end"
    )
    args = parser.parse_args()

    if not os.path.exists(args.logfile):
        print(f"File not found: {args.logfile}")
        sys.exit(1)

    analyze_text_log(args.logfile, discard=args.discard)


if __name__ == "__main__":
    main()
