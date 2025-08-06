# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from log_parser import parse_log_as_paths_latencies

np.set_printoptions(precision=2)

linestyles = ["-", "--", ":", "-."]
colors = ["blue", "red", "green", "purple", "orange", "pink", "brown"]
index = 0

# This python function parses the data-flow-tracking log file
# The format is the following:
# (replayer,1685129021110968,1685129021112852) -> (format_converter,1685129021113053,1685129021159460)
#                                   -> (lstm_inferer,1685129021159626,1685129021161404)
#                                   -> (tool_tracking_postprocessor,1685129021161568,1685129021194271)
#                                   -> (holoviz,1685129021194404,1685129021265517)


# This function merges the latencies of the same path from different log files
def merge_path_latencies(multiple_path_latencies, skip_begin_messages=10, discard_last_messages=10):
    merged_path_latencies = {}
    for path_latencies in multiple_path_latencies:
        for path, latencies in path_latencies.items():
            modified_latencies = latencies[skip_begin_messages:-discard_last_messages]
            if path in merged_path_latencies:
                merged_path_latencies[path].extend(modified_latencies)
            else:
                merged_path_latencies[path] = modified_latencies
    return merged_path_latencies


# Creates a CDF from the provided latencies
def get_cdf_data(latencies):
    data = sorted(latencies)
    n = len(data)
    p = []
    for i in range(n):
        p.append(i / n)
    return data, p


# draw a CDF curve of the latencies using matplotlib where Y-Axis is the CDF and X-Axis is the
# latency
def draw_cdf(ax, latencies, label=None):
    global index

    data, p = get_cdf_data(latencies)

    colorindex = index % len(colors)
    linestylesindex = index % len(linestyles)
    ax.plot(
        data,
        p,
        label=label,
        linewidth=2.0,
        color=colors[colorindex],
        linestyle=linestyles[linestylesindex],
    )
    index += 1


def init_cdf_plot(title=None):
    fig, ax = plt.subplots()
    ax.set_xlabel("End-to-End Latency (ms)")
    ax.set_ylabel("CDF")
    if title:
        ax.set_title(title)
    return fig, ax


def complete_cdf_plot(fig, ax, operator_legends=None):
    ax.grid(True, axis="y")
    vals = ax.get_yticks()
    # convert the Y-axis ticks to percentage
    ax.set_yticks(vals)
    ax.set_yticklabels(["{:,.0%}".format(x) for x in vals])
    ax.set_ylim([-0.05, 1.05])
    # ax.legend(prop={'size': 12}, loc="best")
    legends = ax.legend(prop={"size": 12}, loc="upper center", ncol=2)
    bbox_yoffset = 0.12 * len(legends.get_texts()) / (2 if len(legends.get_texts()) > 2 else 1)
    bbox_to_anchor = (0.5, 1 + bbox_yoffset)
    legends.set_bbox_to_anchor(bbox_to_anchor)
    # also show operator legends in a separate box above the legends
    if operator_legends:
        operator_legends_str = "operator name legends:\n"
        for legend, operator in operator_legends.items():
            operator_legends_str += legend + ": " + operator + "\n"
        ax.text(
            0,
            1 + bbox_yoffset + 0.15,
            operator_legends_str,
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()


def latency_percentile(latencies, percentile, is_sorted=False):
    if not is_sorted:
        latencies = sorted(latencies)
    if percentile == 100:
        return latencies[-1]
    index = int(len(latencies) * percentile / 100.0)
    return latencies[index]


def get_latency_difference(latencies, percentile_start, percentile_end):
    data = sorted(latencies)
    start_value = latency_percentile(data, percentile_start, is_sorted=True)
    end_value = latency_percentile(data, percentile_end, is_sorted=True)
    return "{:.2f}".format(end_value - start_value)


# This function shortens a path by taking first 3 letters of each operator name if
# it's more than 3 letters long
def shorten_path(path, operator_legends, path_separator="→ "):
    operators = path.split(path_separator)
    modified_operators = []
    for operator in operators:
        modified_operator_name = operator[:3] if len(operator) > 3 else operator
        if modified_operator_name not in operator_legends:
            operator_legends[modified_operator_name] = operator
        else:
            if operator_legends[modified_operator_name] != operator:
                print(
                    f"\033[91mERROR: Operator {operator} has the same first 3 letters\
                      as {operator_legends[modified_operator_name]}\033[0m"
                )
                print(
                    "\033[91mCDF Curve legends for operators cannot be created. \
                    CDF Curve creation aborted.\033[0m"
                )
                sys.exit(1)
        modified_operators.append(modified_operator_name)
    return path_separator.join(modified_operators)


# print metric title in a green background with 60 "=" before and after the title
def print_metric_title(title):
    print("\n\033[42m" + "============================================================" + "\033[0m")
    # center align title according to its length with respect to 60 columns
    print("\033[42m" + title.center(60) + "\033[0m")
    print("\033[42m" + "============================================================" + "\033[0m")


def print_group_name_with_log_files(group_name, log_files):
    # print group name in blue color font
    # print log files in grey color font
    print(
        "\n\033[94m"
        + "Group: \033[1m"
        + group_name
        + "\033[0m \033[90m"
        + "("
        + ", ".join(log_files)
        + ")\033[0m"
    )
    print("--------------------")


def print_path_metric_ms(path, metric_ms):
    # print path in blue background
    # print metric_ms in bold and blue foregoround color
    print("\033[1mPath:" + "\033[0m " + path + ": \033[1m\033[94m" + str(metric_ms) + " ms\033[0m")


def print_metric(metric_title, metric_value):
    print("\033[1m" + metric_title + "\033[0m: \033[1m\033[94m" + str(metric_value) + "\033[0m")


# write a main function that takes a log file as argument and calls parse line
def main():
    parser = argparse.ArgumentParser(
        description="Analyze the log files generated by the Data Frame Flow Tracking\
                     module in Holoscan SDK"
    )

    parser.add_argument(
        "-m", "--max", action="store_true", help="show the maximum latencies for all paths"
    )

    parser.add_argument(
        "-a", "--avg", action="store_true", help="show the average latencies for all paths"
    )

    parser.add_argument(
        "--median", action="store_true", help="show the median latencies for all paths"
    )

    parser.add_argument(
        "--stddev",
        action="store_true",
        help="show the standard deviation of latencies for all paths",
    )

    parser.add_argument(
        "--min", action="store_true", help="show the minimum latencies for all paths"
    )

    parser.add_argument(
        "--tail",
        action="store_true",
        help="show the difference between 95 and 100 percentile latencies\
              (latency distribution tail) for all paths",
    )

    parser.add_argument(
        "--flatness",
        action="store_true",
        help="show the difference between 10 and 90 percentile latencies\
              (latency distribution flatness) for all paths",
    )

    parser.add_argument(
        "-p",
        "--percentile",
        nargs="+",
        type=float,
        help="provide a list of percentile values (e.g., '90 95 99 99.9').\
              It will display these percentile latencies for all paths.",
        required=False,
    )

    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="save the respective values (max, avg, median, etc.) of the first path for every group\
              in a CSV file in comma-separated format. (avg: avg_values.csv, max: max_values.csv, \
              gpu utilization: avg_gpu_utilization_values.csv)",
        required=False,
    )

    parser.add_argument(
        "--cdash",
        action="store_true",
        help="write out the values for CTest/CDash",
        required=False,
    )

    parser.add_argument(
        "--draw-cdf",
        nargs="?",
        type=str,
        const="cdf_curve.png",
        help="draw an end-to-end latency CDF curve for the first path of each group of log files.\
              An (optional) filename could also be provided where the graph will be saved.",
        required=False,
    )

    parser.add_argument(
        "--draw-cdf-paths",
        nargs="?",
        type=str,
        const="cdf_curve_paths.png",
        help="draw an end-to-end latency CDF curve for the every path in each group of log files.\
              An (optional) filename could also be provided where the graph will be saved.",
        required=False,
    )

    parser.add_argument(
        "--no-display-graphs",
        action="store_true",
        help="don't display the graphs in a window. Graphs are displayed by default with --draw-cdf\
              and --draw-cdf-paths options.",
        default=False,
        required=False,
    )

    parser.add_argument(
        "-u",
        "--group-utilization-files",
        nargs="+",
        action="append",
        help="specify a group of the GPU utilization files to combine and analyze.\
              You can optionally specify a group name at the end of the list of utilization files",
        required=False,
    )

    requiredArgument = parser.add_argument_group("required arguments")
    requiredArgument.add_argument(
        "-g",
        "--group-log-files",
        nargs="+",
        action="append",
        help="specify a group of the log files to combine and analyze.\
              You can optionally specify a group name at the end of the list of log files",
        required=True,
    )

    args = parser.parse_args()

    # Tell CTest to send the full output to CDash
    if args.cdash:
        print("CTEST_FULL_OUTPUT")

    # Group the log files and parse the latencies from the log files
    groups = args.group_log_files

    group_name = "Group"
    group_name_counter = 1

    grouped_path_latenices = {}
    grouped_log_files = {}
    for group in groups:
        # check whether the last entry has a dot in it, then it does not have a group name
        current_group_name = ""
        current_log_files = []
        if group[-1].find(".") != -1:
            current_group_name = group_name + str(group_name_counter)
            group_name_counter += 1
            current_log_files = group
        else:
            current_group_name = group[-1]
            current_log_files = group[:-1]
        if len(current_log_files) == 0:
            print(
                "\033[91mError: No log files provided for group: " + current_group_name + "\033[0m"
            )
            sys.exit(1)
        parsed_latencies_per_file = []
        for log_file in current_log_files:
            parsed_latencies_per_file.append(parse_log_as_paths_latencies(log_file))
        grouped_path_latenices[current_group_name] = merge_path_latencies(parsed_latencies_per_file)
        grouped_log_files[current_group_name] = current_log_files

    if args.max:
        if args.save_csv:
            with open("max_values.csv", "w") as f:
                f.truncate(0)
        print_metric_title("Maximum (Worst-case) Latencies")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print_group_name_with_log_files(group_name, grouped_log_files[group_name])
            for path, latency in paths_latencies.items():
                print_path_metric_ms(path, str(round(np.max(latency), 2)))
            (path, latency) = next(iter(paths_latencies.items()))
            if args.cdash:
                print(
                    f'<CTestMeasurement type="numeric/double" name="maximum_latency_{group_name}">'
                    + str(round(np.max(latency), 2))
                    + "</CTestMeasurement>"
                )
            if args.save_csv:
                with open("max_values.csv", "a") as f:
                    f.write(str(round(np.max(latency), 2)) + ",")

    if args.avg:
        if args.save_csv:
            with open("avg_values.csv", "w") as f:
                f.truncate(0)
        print_metric_title("Average Latencies")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print_group_name_with_log_files(group_name, grouped_log_files[group_name])
            for path, latency in paths_latencies.items():
                print_path_metric_ms(path, str(round(np.mean(latency), 2)))
            (path, latency) = next(iter(paths_latencies.items()))
            if args.cdash:
                print(
                    f'<CTestMeasurement type="numeric/double" name="average_latency_{group_name}">'
                    + str(round(np.mean(latency), 2))
                    + "</CTestMeasurement>"
                )
            if args.save_csv:
                with open("avg_values.csv", "a") as f:
                    f.write(str(round(np.mean(latency), 2)) + ",")

    if args.median:
        if args.save_csv:
            with open("median_values.csv", "w") as f:
                f.truncate(0)
        print_metric_title("Median Latencies")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print_group_name_with_log_files(group_name, grouped_log_files[group_name])
            for path, latency in paths_latencies.items():
                print_path_metric_ms(path, str(round(np.median(latency), 2)))
            (path, latency) = next(iter(paths_latencies.items()))
            if args.cdash:
                print(
                    f'<CTestMeasurement type="numeric/double" name="median_latency_{group_name}">'
                    + str(round(np.median(latency), 2))
                    + "</CTestMeasurement>"
                )
            if args.save_csv:
                with open("median_values.csv", "a") as f:
                    f.write(str(round(np.median(latency), 2)) + ",")

    if args.stddev:
        if args.save_csv:
            with open("stddev_values.csv", "w") as f:
                f.truncate(0)
        print_metric_title("Standard Deviation of Latencies")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print_group_name_with_log_files(group_name, grouped_log_files[group_name])
            for path, latency in paths_latencies.items():
                print_path_metric_ms(path, str(round(np.std(latency), 2)))
            (path, latency) = next(iter(paths_latencies.items()))
            if args.cdash:
                print(
                    f'<CTestMeasurement type="numeric/double" name="stddev_latency_{group_name}">'
                    + str(round(np.std(latency), 2))
                    + "</CTestMeasurement>"
                )
            if args.save_csv:
                with open("stddev_values.csv", "a") as f:
                    f.write(str(round(np.std(latency), 2)) + ",")

    if args.min:
        if args.save_csv:
            with open("min_values.csv", "w") as f:
                f.truncate(0)
        print_metric_title("Minimum Latencies")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print_group_name_with_log_files(group_name, grouped_log_files[group_name])
            for path, latency in paths_latencies.items():
                print_path_metric_ms(path, str(round(min(latency), 2)))
            (path, latency) = next(iter(paths_latencies.items()))
            if args.cdash:
                print(
                    f'<CTestMeasurement type="numeric/double" name="min_latency_{group_name}">'
                    + str(round(min(latency), 2))
                    + "</CTestMeasurement>"
                )
            if args.save_csv:
                with open("min_values.csv", "a") as f:
                    f.write(str(round(min(latency), 2)) + ",")

    if args.tail:
        if args.save_csv:
            with open("tail_values.csv", "w") as f:
                f.truncate(0)
        print_metric_title("Latency Distribution Tail (95-100 percentile)")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print_group_name_with_log_files(group_name, grouped_log_files[group_name])
            for path, latency in paths_latencies.items():
                print_path_metric_ms(path, get_latency_difference(latency, 95, 100))
            (path, latency) = next(iter(paths_latencies.items()))
            latency_tail_one_path = str(get_latency_difference(latency, 95, 100))
            if args.cdash:
                print(
                    f'<CTestMeasurement type="numeric/double" name="distribution_tail_{group_name}">'
                    + latency_tail_one_path
                    + "</CTestMeasurement>"
                )
            if args.save_csv:
                with open("tail_values.csv", "a") as f:
                    f.write(latency_tail_one_path + ",")

    if args.flatness:
        if args.save_csv:
            with open("flatness_values.csv", "w") as f:
                f.truncate(0)
        print_metric_title("Latency Distribution Flatness (10-90 percentile)")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print_group_name_with_log_files(group_name, grouped_log_files[group_name])
            for path, latency in paths_latencies.items():
                print_path_metric_ms(path, get_latency_difference(latency, 10, 90))
            (path, latency) = next(iter(paths_latencies.items()))
            latency_flatness_one_path = str(get_latency_difference(latency, 10, 90))
            if args.cdash:
                print(
                    f'<CTestMeasurement type="numeric/double" name="\
                      distribution_flatness_{group_name}">'
                    + latency_flatness_one_path
                    + "</CTestMeasurement>"
                )
            if args.save_csv:
                with open("flatness_values.csv", "a") as f:
                    f.write(latency_flatness_one_path + ",")

    if args.percentile:
        for percentile in args.percentile:
            percentile_file = f"percentile_{percentile}_values.csv"
            with open(percentile_file, "w") as f:
                f.truncate(0)
            print_metric_title(f"Latency Percentile ({percentile})")
            for group_name, paths_latencies in grouped_path_latenices.items():
                print_group_name_with_log_files(group_name, grouped_log_files[group_name])
                for path, latency in paths_latencies.items():
                    latency_percentile_str = "{:.2f}".format(
                        latency_percentile(latency, float(percentile))
                    )
                    print_path_metric_ms(path, latency_percentile_str)
                (path, latency) = next(iter(paths_latencies.items()))
                latency_percentile_filtered_one_path = "{:.2f}".format(
                    latency_percentile(latency, float(percentile))
                )
                if args.cdash:
                    print(
                        f'<CTestMeasurement type="numeric/double" name\
                          ="percentile_{percentile}_{group_name}">'
                        + latency_percentile_filtered_one_path
                        + "</CTestMeasurement>"
                    )
                if args.save_csv:
                    with open(percentile_file, "a") as f:
                        f.write(latency_percentile_filtered_one_path + ",")

    if args.draw_cdf:
        fig, ax = init_cdf_plot()
        for group_name, paths_latencies in grouped_path_latenices.items():
            draw_cdf(ax, paths_latencies[list(paths_latencies.keys())[0]], group_name)
        complete_cdf_plot(fig, ax)
        plt.tight_layout()
        plt.savefig(args.draw_cdf, bbox_inches="tight")
        print("Saved the CDF curve graph of the first path of each group in:", args.draw_cdf)
        if not args.no_display_graphs:
            plt.tight_layout()
            plt.show()
        if args.cdash:
            print(
                '<CTestMeasurementFile type="image/png" name="cdf_plot">'
                + args.draw_cdf
                + "</CTestMeasurementFile>"
            )

    if args.draw_cdf_paths:
        fig, ax = init_cdf_plot()
        operator_legends = {}
        for group_name, paths_latencies in grouped_path_latenices.items():
            for path, latency in paths_latencies.items():
                draw_cdf(ax, latency, group_name + "-" + shorten_path(path, operator_legends))
        complete_cdf_plot(fig, ax, operator_legends=operator_legends)
        plt.tight_layout()
        plt.savefig(args.draw_cdf_paths, bbox_inches="tight")
        print("Saved the CDF curve graph of all paths of each group in:", args.draw_cdf_paths)
        if not args.no_display_graphs:
            plt.tight_layout()
            plt.show()

    if args.group_utilization_files:
        # combine the GPU utilization files the same way as done for log files
        util_groups = args.group_utilization_files
        util_group_name = "UGroup"
        util_group_name_counter = 1
        grouped_gpu_util = {}
        grouped_gpu_util_log_files = {}
        for group in util_groups:
            current_util_group_name = ""
            current_gpu_util_files = []
            if group[-1].find(".") != -1:
                current_util_group_name = util_group_name + str(util_group_name_counter)
                util_group_name_counter += 1
                current_gpu_util_files = group
            else:
                current_util_group_name = group[-1]
                current_gpu_util_files = group[:-1]
            if len(current_gpu_util_files) == 0:
                print(
                    "\033[91mError: No GPU utilization files provided for group: "
                    + current_util_group_name
                    + "\033[0m"
                )
                sys.exit(1)
            parsed_gpu_utils = []
            for gpu_util_file in current_gpu_util_files:
                with open(gpu_util_file, "r") as f:
                    all_utils = f.readlines()
                    for lines in all_utils:
                        for value in lines.strip().split(","):
                            parsed_gpu_utils.append(float(value))
            grouped_gpu_util[current_util_group_name] = parsed_gpu_utils
            grouped_gpu_util_log_files[current_util_group_name] = current_gpu_util_files
        if args.save_csv:
            with open("avg_gpu_utilization_values.csv", "w") as f:
                f.truncate(0)
        print_metric_title("Average GPU Utilization")
        for group_name, gpu_utils in grouped_gpu_util.items():
            print_group_name_with_log_files(group_name, grouped_gpu_util_log_files[group_name])
            print_metric("Average GPU Utilization", str(round(np.mean(gpu_utils), 2)) + "%")
            if args.save_csv:
                with open("avg_gpu_utilization_values.csv", "a") as f:
                    f.write(str(round(np.mean(gpu_utils), 2)) + ",")


if __name__ == "__main__":
    main()
