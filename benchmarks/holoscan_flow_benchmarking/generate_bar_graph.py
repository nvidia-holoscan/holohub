# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import datetime
import os

import matplotlib.pyplot as plt

keyword_dictionary = {
    "min": "Minimum E2E Latency (ms)",
    "median": "Median E2E Latency (ms)",
    "max": "Maximum E2E Latency (ms)",
    "tail": "Latency Tail (ms)",
    "flatness": "Latency Flatness (ms)",
    "avg": "Average E2E Latency (ms)",
    "stddev": "Latency Stddev (ms)",
    "percentile": "Percentile Latency (ms)",
}

long_keyword_dictionary = {
    "min": "Minimum End-to-end Latency (ms)",
    "median": "Median End-to-end Latency (ms)",
    "max": "Maximum End-to-end Latency (ms)",
    "tail": "Latency Distribution Tail (ms)",
    "flatness": "Latency Distribution Flatness (ms)",
    "avg": "Average End-to-end Latency (ms)",
    "stddev": "Latency Standard Deviation (ms)",
    "percentile": "Percentile Latency (ms)",
}


def keyword_to_title(keyword, long=0):
    return keyword_dictionary[keyword] if long == 0 else long_keyword_dictionary[keyword]


# Parse a CSV file so that each line is an array of float numbers
def parse_csv_file(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    data = [line.strip().rstrip().split(",") for line in data]
    # if value is not whitespace
    data = [[float(num) for num in line if num != ""] for line in data]
    return data if len(data) > 1 else data[0]


def main():
    CSV_FILENAME_DELIMITER = "_"

    parser = argparse.ArgumentParser(
        description="Generate bar graphs from a CSV file of latency statistics generated with `analyze.py`."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the latency statistics CSV file. "
        "The CSV filename must follow the `analyze.py` output convention `<statistics_name>_values.csv`."
        "Assumes one row of data with each column representing an increasing number of instances. ",
    )
    parser.add_argument(
        "--output_extension", default="png", help="Output file extension (default: png)"
    )
    parser.add_argument("--app", default="endoscopy", help="Application name for file output")
    parser.add_argument(
        "--title", default="Endoscopy Tool Tracking {current_time}", help="Graph title"
    )
    parser.add_argument("--quiet", default=False, action="store_true", help="Suppress output")
    args = parser.parse_args()

    # Use the parsed arguments
    csv_file = args.csv_file
    output_extension = args.output_extension

    # Parse the CSV file
    data = parse_csv_file(csv_file)
    instances = [x for x in range(1, len(data) + 1)]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    bar_width = 0.60

    # Function to add bars to the plot
    def add_bars(values, position, color, hatch, label=""):
        xticks = [x + position for x in instances]
        bars = ax.bar(xticks, values, bar_width, color=color, alpha=0.5, label=label, hatch=hatch)
        for bar, v in zip(bars, values):
            if v != "N/A":
                height = bar.get_height()
                ax.annotate(
                    "{}".format(v),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )
        return xticks

    all_ticks = []
    ticks = add_bars(data, 0, "tab:blue", None, None)
    all_ticks.append(ticks)

    ax.set_xlabel("Number of Instances", fontsize=14, fontweight="bold")

    csv_filename_components = os.path.basename(csv_file).split(CSV_FILENAME_DELIMITER)
    keyword = csv_filename_components[0]
    if keyword == "percentile":
        percentile_value = csv_filename_components[1]
    yaxis_label = (
        keyword_to_title(keyword)
        if keyword != "percentile"
        else percentile_value + " " + keyword_to_title(keyword)
    )
    ax.set_ylabel(yaxis_label, fontsize=14, fontweight="bold")

    xaxis_ticks = (
        [(first + last) / 2 for first, last in zip(all_ticks[0], all_ticks[-1])]
        if len(all_ticks) > 1
        else all_ticks[0]
    )
    ax.set_xticks(xaxis_ticks)
    ax.set_xticklabels(instances, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim([0, max(data) * 1.2])
    current_time = datetime.datetime.now().strftime("%m/%d/%Y %H:%M")
    title = args.title.format(current_time=current_time)
    ax.set_title(
        f"{title}",
        fontsize=14,
        pad=14,
    )

    plt.tight_layout()
    output_keyword = "".join(csv_filename_components[:-1])
    output_file = f"{args.app}_{output_keyword}.{output_extension}"
    plt.savefig(output_file, bbox_inches="tight")
    if not args.quiet:
        print(
            f'<CTestMeasurementFile type="image/png" name="instances_{output_keyword}">'
            + output_file
            + "</CTestMeasurementFile>"
        )


if __name__ == "__main__":
    main()
