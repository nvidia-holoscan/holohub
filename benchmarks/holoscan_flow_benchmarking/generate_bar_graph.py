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


import datetime
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

keyword_dictionary = {
    "min": "Minimum E2E Latency",
    "median": "Median E2E Latency",
    "max": "Maximum E2E Latency",
    "tail": "Latency Tail",
    "flatness": "Latency Flatness",
    "avg": "Average E2E Latency",
    "stddev": "Latency Stddev",
    "percentile": "Percentile Latency",
}

long_keyword_dictionary = {
    "min": "Minimum End-to-end Latency",
    "median": "Median End-to-end Latency",
    "max": "Maximum End-to-end Latency",
    "tail": "Latency Distribution Tail",
    "flatness": "Latency Distribution Flatness",
    "avg": "Average End-to-end Latency",
    "stddev": "Latency Standard Deviation",
    "percentile": "Percentile Latency",
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
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("Usage: python generate_bar_graphs.py <csv_file> <output_extension>")
        exit(1)
    output_extension = "png"
    if len(sys.argv) > 2:
        output_extension = sys.argv[2]

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

    keyword = csv_file.split("_")[0]
    if keyword == "percentile":
        percentile_value = csv_file.split("_")[1]
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
    ax.set_title(
        f"Endoscopy Tool Tracking ({current_time})",
        fontsize=14,
        pad=14,
    )

    plt.tight_layout()
    output_keyword = csv_file[: csv_file.rfind("_")]
    output_file = f"endoscopy_{output_keyword}.{output_extension}"
    plt.savefig(output_file, bbox_inches="tight")
    print(
        f'<CTestMeasurementFile type="image/png" name="instances_{output_keyword}">'
        + output_file
        + "</CTestMeasurementFile>"
    )


if __name__ == "__main__":
    main()
