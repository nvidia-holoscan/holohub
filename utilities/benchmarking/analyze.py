# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

np.set_printoptions(precision=2)

linestyles = ['-', '--', ':', '-.']
colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink', 'brown']
index = 0

def parse_line(line):
    operators = line.split("->")
    # print (operators)
    op_timestamps = []
    for operator in operators:
        # trim whitespaces for left and right side
        # print ("op: ", operator)
        op_name_timestamp = operator.strip().rstrip()[1:-1]
        op_timestamps.append(op_name_timestamp.split(","))
    return op_timestamps

# return a path and latency pair where a path is a comma separate string of operators
def get_latency(op_timestamps):
    path = ""
    latency = 0
    for op_timestamp in op_timestamps:
        # print (op_timestamp)
        path += op_timestamp[0] + ","
    # convert the latency to ms
    latency = float(int(op_timestamps[-1][2]) - int(op_timestamps[0][1])) / 1000
    return path[:-1], latency

# This python function parses the DFFT-generated log file
# The format is the following:
# (replayer,1685129021110968,1685129021112852) -> (format_converter,1685129021113053,1685129021159460) -> (lstm_inferer,1685129021159626,1685129021161404) -> (tool_tracking_postprocessor,1685129021161568,1685129021194271) -> (holoviz,1685129021194404,1685129021265517)

# The format is (Operator1, receive timestamp, publish timestsamp) -> (Operator2, receive timestamp,
# publish timestsamp) -> ... -> (OperatorN, receive timestamp, publish timestsamp)
def parse_log(log_file):
    with open(log_file, "r") as f:
        paths_latencies = {}
        for line in f:
            # print ("line: ", line)
            if line[0] == "(":
                path_latency = get_latency(parse_line(line))
                if path_latency[0] in paths_latencies:
                    paths_latencies[path_latency[0]].append(path_latency[1])
                else:
                    paths_latencies[path_latency[0]] = [path_latency[1]]
        return paths_latencies

# This function merges the latencies of the same path from different log files
def merge_path_latencies(multiple_path_latencies, skip_begin_messages = 10, discard_last_messages = 10):
    merged_path_latencies = {}
    for path_latencies in multiple_path_latencies:
        for path, latencies in path_latencies.items():
            modified_latencies = latencies[skip_begin_messages:-discard_last_messages]
            if path in merged_path_latencies:
                merged_path_latencies[path].extend(modified_latencies)
            else:
                merged_path_latencies[path] = modified_latencies
    return merged_path_latencies

def get_avg_latencies(paths_latencies, skip_begin_messages = 10, discard_last_messages = 10):
    avg_latencies = {}
    for path in paths_latencies:
        avg_latencies[path] = round(np.mean(paths_latencies[path][skip_begin_messages:-discard_last_messages]), 2)
    return avg_latencies

def get_max_latencies(paths_latencies, skip_begin_messages = 10, discard_last_messages = 10):
    max_latencies = {}
    for path in paths_latencies:
        max_latencies[path] = max(paths_latencies[path][skip_begin_messages:-discard_last_messages])
    return max_latencies

def get_cdf_data(latencies):
    data = sorted(latencies)
    n = len(data)
    p = []
    for i in range(n):
        p.append(i/n)
    return data, p

# draw a CDF curve of the latencies using matplotlib where Y-Axis is the CDF and X-Axis is the
# latency
def draw_cdf(ax, latencies, label = None):
    global index

    data, p = get_cdf_data(latencies)
    data_max = max(data)
    data_avg = np.mean(data)
    data_stddev = np.std(data)

    colorindex = index % len(colors)
    linestylesindex = index % len(linestyles)
    ax.plot(data, p, label=label, linewidth=2.0, color=colors[colorindex], linestyle=linestyles[linestylesindex])
    ax.axvline(x=data_avg, color=colors[colorindex], linestyle=linestyles[linestylesindex], linewidth=1)
    # put a shaded area of stddev around average latency
    ax.axvspan(data_avg - data_stddev, data_avg + data_stddev, alpha=0.2, color=colors[colorindex])

    ax.axvline(x=data_max, color=colors[colorindex], linestyle=linestyles[linestylesindex], linewidth=1.5)
    index += 1

def init_cdf_plot(title=None):
    fig, ax = plt.subplots()
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    if title:
        ax.set_title(title)
    return fig, ax

def complete_cdf_plot(fig, ax):
    # ax.set_ylim([-0.02, 1.03])
    vals = ax.get_yticks()
    # convert the Y-axis ticks to percentage
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # ax.legend(prop={'size': 12}, loc="best")
    legends = ax.legend(prop={'size': 12}, loc="upper center", ncol=2)
    bbox_to_anchor=(0.5, 1 + 0.12 * len(legends.get_texts()) / (2 if len(legends.get_texts()) > 2 else 1))
    legends.set_bbox_to_anchor(bbox_to_anchor)
    fig.tight_layout()

def get_latency_difference(latencies, percentile_start, percentile_end, skip_begin_messages = 10, discard_last_messages = 10):
    data = sorted(latencies[skip_begin_messages:-discard_last_messages])
    n = len(data)
    start_index = int(n * percentile_start/100.0)
    end_index = int(n * percentile_end/100.0 - 1)
    return "{:.2f}".format(data[end_index] - data[start_index])

# This function shortens a path by taking first 3 letters of each operator name if
# it's more than 4 letters long
def shorten_path(path):
    operators = path.split(",")
    modified_operators = []
    for operator in operators:
        if len(operator) > 4:
            modified_operators.append(operator[:3])
        else:
            modified_operators.append( operator)
    return ",".join(modified_operators)

# write a main function that takes a log file as argument and calls parse line
def main():
    parser = argparse.ArgumentParser(description='Analyze the log files generated by the Data Frame Flow Tracking module in Holoscan SDK')

    parser.add_argument("-m", "--max", action="store_true", help="show the maximum latencies for all paths")

    parser.add_argument("-a", "--avg", action="store_true", help="show the average latencies for all paths")

    parser.add_argument("--median", action="store_true", help="show the median latencies for all paths")

    parser.add_argument("--stddev", action="store_true", help="show the standard deviation of latencies for all paths")

    parser.add_argument("--min", action="store_true", help="show the minimum latencies for all paths")

    parser.add_argument("--tail", action="store_true", help="show the difference between 95 and 100 percentile latencies for all paths")

    parser.add_argument("--flatness", action="store_true", help="show the difference between 10 and 90 percentile latencies for all paths")

    parser.add_argument("--save-csv", action="store_true", help="save the respective values (max, avg, median, etc.) of the first path for every group in a CSV file in comma-separated format. (avg: avg_values.csv, max:max_values.csv)", required=False)

    parser.add_argument("--draw-cdf", nargs="?", type=str, const="cdf_curve.png", help="draw a end-to-end latency CDF curve for the first path of each group of log files. An (optional) filename could also be provided in which the graph will be saved.", required=False)

    parser.add_argument("--draw-cdf-paths", nargs="?", type=str, const="cdf_curve_paths.png", help="draw a end-to-end latency CDF curve for the every path in each group of log files. An (optional) filename could also be provided in which the graph will be saved.", required=False)

    parser.add_argument("--display-graphs", action="store_true", help="display the graphs", default=False, required=False)

    parser.add_argument("-u", "--group-utilization-files", nargs="+", action="append", help="specify a group of the GPU utilization files to combine and analyze. You can optionally specify a group name at the end of the list of utilization files", required=False)

    requiredArgument = parser.add_argument_group('required arguments')
    requiredArgument.add_argument("-g", "--group_log_files", nargs="+", action="append", help="specify a group of the log files to combine and analyze. You can optionally specify a group name at the end of the list of log files", required=True)

    args = parser.parse_args()

    bar_graph_options = ["avg", "max"]

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
        parsed_latencies_per_file = []
        for log_file in current_log_files:
            parsed_latencies_per_file.append(parse_log(log_file))
        grouped_path_latenices[current_group_name] = merge_path_latencies(parsed_latencies_per_file)
        grouped_log_files[current_group_name] = current_log_files

    if args.max:
        if args.save_csv:
            with open("max_values.csv", "w") as f:
                f.truncate(0)
        print ("==================\nMax Latencies\n==============")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print ("Group : ", group_name, "(", ", ".join(grouped_log_files[group_name]), ")" "\n------------------")
            max_latencies = get_max_latencies(paths_latencies)
            for path, latency in max_latencies.items():
                print ("Path:", path, ": ", latency, "ms")
            if args.save_csv:
                with open("max_values.csv", "a") as f:
                    f.write(str(max_latencies[list(max_latencies.keys())[0]]) + ",")
    if args.avg:
        if args.save_csv:
            with open("avg_values.csv", "w") as f:
                f.truncate(0)
        print ("==================\nAverage Latencies\n==================")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print ("Group : ", group_name, "(", ", ".join(grouped_log_files[group_name]), ")" "\n------------------")
            avg_latencies = get_avg_latencies(paths_latencies)
            for path, latency in avg_latencies.items():
                print ("Path:", path, ": ", latency, "ms")
            if args.save_csv:
                with open("avg_values.csv", "a") as f:
                    f.write(str(avg_latencies[list(avg_latencies.keys())[0]]) + ",")

    if args.median:
        if args.save_csv:
            with open("median_values.csv", "w") as f:
                f.truncate(0)
        print ("==================\nMedian Latencies\n==================")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print ("Group : ", group_name, "(", ", ".join(grouped_log_files[group_name]), ")" "\n------------------")
            for path, latency in paths_latencies.items():
                print ("Path:", path, ": ", np.median(latency), "ms")
            if args.save_csv:
                with open("median_values.csv", "a") as f:
                    f.write(str(np.median(paths_latencies[list(paths_latencies.keys())[0]])) + ",")

    if args.stddev:
        if args.save_csv:
            with open("stddev_values.csv", "w") as f:
                f.truncate(0)
        print ("==================\nStandard Deviation of Latencies\n==================")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print ("Group : ", group_name, "(", ", ".join(grouped_log_files[group_name]), ")" "\n------------------")
            for path, latency in paths_latencies.items():
                print ("Path:", path, ": ", np.std(latency), "ms")
            if args.save_csv:
                with open("stddev_values.csv", "a") as f:
                    f.write(str(round(np.std(paths_latencies[list(paths_latencies.keys())[0]]), 2)) + ",")

    if args.min:
        if args.save_csv:
            with open("min_values.csv", "w") as f:
                f.truncate(0)
        print ("==================\nMin Latencies\n==================")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print ("Group : ", group_name, "(", ", ".join(grouped_log_files[group_name]), ")" "\n------------------")
            for path, latency in paths_latencies.items():
                print ("Path:", path, ": ", min(latency), "ms")
            if args.save_csv:
                with open("min_values.csv", "a") as f:
                    f.write(str(min(paths_latencies[list(paths_latencies.keys())[0]])) + ",")

    if args.tail:
        if args.save_csv:
            with open("tail_values.csv", "w") as f:
                f.truncate(0)
        print ("==================\nLatency CDF Curve Tail (95-100)\n==================")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print ("Group : ", group_name, "(", ", ".join(grouped_log_files[group_name]), ")" "\n------------------")
            for path, latency in paths_latencies.items():
                print ("Path:", path, ": ", get_latency_difference(latency, 95, 100), "ms")
            if args.save_csv:
                with open("tail_values.csv", "a") as f:
                    f.write(str(get_latency_difference(paths_latencies[list(paths_latencies.keys())[0]], 95, 100)) + ",")

    if args.flatness:
        if args.save_csv:
            with open("flatness_values.csv", "w") as f:
                f.truncate(0)
        print ("==================\nLatency CDF Curve Flatness (10-90)\n==================")
        for group_name, paths_latencies in grouped_path_latenices.items():
            print ("Group : ", group_name, "(", ", ".join(grouped_log_files[group_name]), ")" "\n------------------")
            for path, latency in paths_latencies.items():
                print ("Path:", path, ": ", get_latency_difference(latency, 10, 90), "ms")
            if args.save_csv:
                with open("flatness_values.csv", "a") as f:
                    f.write(str(get_latency_difference(paths_latencies[list(paths_latencies.keys())[0]], 10, 90)) + ",")

    if args.draw_cdf:
        fig, ax = init_cdf_plot()
        for group_name, paths_latencies in grouped_path_latenices.items():
            draw_cdf(ax, paths_latencies[list(paths_latencies.keys())[0]], group_name)
        complete_cdf_plot(fig, ax)
        plt.tight_layout()
        plt.savefig(args.draw_cdf, bbox_inches='tight')
        if args.display_graphs:
            plt.show()

    if args.draw_cdf_paths:
        fig, ax = init_cdf_plot()
        for group_name, paths_latencies in grouped_path_latenices.items():
            for path, latency in paths_latencies.items():
                draw_cdf(ax, latency, group_name + "-" + shorten_path(path))
        complete_cdf_plot(fig, ax)
        plt.tight_layout()
        plt.savefig(args.draw_cdf_paths, bbox_inches='tight')
        if args.display_graphs:
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
        print ("==================\nAverage GPU Utilization\n==================")
        for group_name, gpu_utils in grouped_gpu_util.items():
            print ("Group : ", group_name, "(", ", ".join(grouped_gpu_util_log_files[group_name]), ")" "\n------------------")
            print ("Average GPU Utilization: ", round(np.mean(gpu_utils), 2), "%")
            if args.save_csv:
                with open("avg_gpu_utilization_values.csv", "a") as f:
                    f.write(str(round(np.mean(gpu_utils), 2)) + ",")


if __name__ == "__main__":
    main()