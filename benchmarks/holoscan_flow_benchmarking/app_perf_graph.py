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
import os
import queue
import sys
import time

import pydot
from log_parser import parse_line_from_log

# independent run instruction: python3 app_perf_graph.py <filenames>


# This function returns the latency between operator pairs and the time spent in each operator
def get_operator_latency(op_timestamps):
    operator_latency = {}
    edge_latency = {}
    for i in range(len(op_timestamps)):
        operator_latency[op_timestamps[i][0]] = (
            float(int(op_timestamps[i][2]) - int(op_timestamps[i][1])) / 1000.0
        )
        if i < len(op_timestamps) - 1:
            edge_latency[(op_timestamps[i][0], op_timestamps[i + 1][0])] = (
                float(int(op_timestamps[i + 1][1]) - int(op_timestamps[i][2])) / 1000.0
            )
    return operator_latency, edge_latency


def update_op_edge_latency(
    op_latency,
    edge_latency,
    operator_avg_latencies,
    operator_max_latencies,
    edge_avg_latencies,
    edge_max_latencies,
    num_samples,
):
    for op, latency in op_latency.items():
        if op in operator_avg_latencies:
            operator_avg_latencies[op] = float(
                operator_avg_latencies[op] * num_samples + latency
            ) / (num_samples + 1)
            operator_max_latencies[op] = max(operator_max_latencies[op], latency)
        else:
            operator_avg_latencies[op] = latency
            operator_max_latencies[op] = latency
    for edge, latency in edge_latency.items():
        if edge in edge_avg_latencies:
            edge_avg_latencies[edge] = float(edge_avg_latencies[edge] * num_samples + latency) / (
                num_samples + 1
            )
            edge_max_latencies[edge] = max(edge_max_latencies[edge], latency)
        else:
            edge_avg_latencies[edge] = latency
            edge_max_latencies[edge] = latency
    return operator_avg_latencies, operator_max_latencies, edge_avg_latencies, edge_max_latencies


def parse_log(
    log_file,
    startline=0,
    operator_avg_latencies={},
    operator_max_latencies={},
    edge_avg_latencies={},
    edge_max_latencies={},
    num_samples=0,
    skip_begin_messages=10,
    discard_last_messages=10,
    livefile=False,
):
    with open(log_file, "r") as f:
        read_lines = 0
        if startline:
            # skip startline number of lines in f
            for _ in range(startline):
                f.readline()
        # this is a buffer to see whether to discard the last 10 messages
        buffered_op_edge_latencies = queue.Queue(maxsize=discard_last_messages + 1)
        for line in f:
            read_lines += 1
            if line[0] == "(":
                if skip_begin_messages > 0 and startline == 0:
                    skip_begin_messages -= 1
                    continue
                op_latency, edge_latency = get_operator_latency(parse_line_from_log(line))
                buffered_op_edge_latencies.put((op_latency, edge_latency))
                if buffered_op_edge_latencies.full():
                    old_op_latency, old_edge_latency = buffered_op_edge_latencies.get()
                    update_op_edge_latency(
                        old_op_latency,
                        old_edge_latency,
                        operator_avg_latencies,
                        operator_max_latencies,
                        edge_avg_latencies,
                        edge_max_latencies,
                        num_samples,
                    )
                    num_samples += 1
        # if the file is live, then we don't want to discard the last few messages
        while (livefile and not buffered_op_edge_latencies.empty()) or (
            not livefile and buffered_op_edge_latencies.qsize() > discard_last_messages
        ):
            old_op_latency, old_edge_latency = buffered_op_edge_latencies.get()
            update_op_edge_latency(
                old_op_latency,
                old_edge_latency,
                operator_avg_latencies,
                operator_max_latencies,
                edge_avg_latencies,
                edge_max_latencies,
                num_samples,
            )
            num_samples += 1
        return (
            operator_avg_latencies,
            operator_max_latencies,
            edge_avg_latencies,
            edge_max_latencies,
            num_samples,
            read_lines,
        )


def create_graph(
    operator_avg_latencies,
    operator_max_latencies,
    edge_avg_latencies,
    edge_max_latencies,
    num_samples,
    highlight,
):
    # Go through all of the operator's average latencies and find the average
    if highlight and len(operator_avg_latencies) > 0:
        values = operator_avg_latencies.values()
        total_sum = sum(values)
        avg_op_latency = total_sum / len(values)

    graph = pydot.Dot(graph_type="digraph")
    for op, latency in operator_avg_latencies.items():
        color = "red" if highlight and (latency > avg_op_latency) else "black"
        penwidth = "2.0" if highlight and (latency > avg_op_latency) else "1.0"
        node = pydot.Node(
            op,
            color=color,
            penwidth=penwidth,
            label="{}\navg: {:.2f}\nmax: {:.2f}".format(op, latency, operator_max_latencies[op]),
        )
        graph.add_node(node)
    for edge, latency in edge_avg_latencies.items():
        edge = pydot.Edge(
            edge[0],
            edge[1],
            label="avg: {:.2f}\nmax: {:.2f}".format(latency, edge_max_latencies[edge]),
        )
        graph.add_edge(edge)
    return graph


# add graph labels
def add_graph_labels(graph, num_samples):
    graph.set_label(
        "Application Performance Graph (latency in ms)\n\
        Number of messages at sink: {}".format(
            num_samples
        )
    )
    graph.set_fontname("Arial")
    graph.set_fontsize(20)
    graph.set_fontcolor("black")
    graph.set_rankdir("TB")
    return graph


def parse_arguments():
    parser = argparse.ArgumentParser(description="Show the application graph with latency data.")
    parser.add_argument(
        "filenames", nargs="+", help="The files to plot. In live mode, provide a folder."
    )
    parser.add_argument(
        "-h",
        "--highlight",
        action="store_true",
        help="highlight nodes in the graph whose average latency is higher than the average of all the operators",
    )
    parser.add_argument(
        "-l", "--live", action="store_true", help="live mode: keep updating the graph with new data"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output file to save the graph to.",
        default="application_perf.dot",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode, prints more informative messages.",
    )

    args = parser.parse_args()

    if not args.live:
        filenames = args.filenames
        if len(filenames) == 0:
            print("Error: No filenames provided")
            parser.print_help()
            sys.exit()
        # check if the files exist
        for filename in filenames:
            if not os.path.isfile(filename):
                print("Error: File {} does not exist or is not a file.".format(filename))
                sys.exit()
    else:
        directory = args.filenames[0]
        print(
            "In live mode. The program will keep updating the graph with new \
                performance data in the provided folder {}.\
                Press Ctrl+C to stop.".format(
                directory
            )
        )
        if len(args.filenames) > 1:
            print(
                "\033[91mfilenames arguments has {} values. In live mode,\
                    only one folder is acceptable. Provide one folder as filenames\
                    in live mode. Exiting.\033[0m".format(
                    len(args.filenames)
                )
            )
            parser.print_help()
            sys.exit()
        # check if directory is a folder
        if not os.path.isdir(directory):
            print(
                "\033[91mThe folder {} does not exist or is not a folder.\
                    In live mode, a folder needs to be provided. Exiting.\033[0m".format(
                    directory
                )
            )
            parser.print_help()
            sys.exit()

    return args


def create_dot_file(filenames, output_filename, live_graph, verbose, highlight):
    if not live_graph:
        filenames = filenames
        operator_avg_latencies = {}
        operator_max_latencies = {}
        edge_avg_latencies = {}
        edge_max_latencies = {}
        num_samples = 0
        for filename in filenames:
            (
                operator_avg_latencies,
                operator_max_latencies,
                edge_avg_latencies,
                edge_max_latencies,
                num_samples,
                read_lines,
            ) = parse_log(
                filename,
                0,
                operator_avg_latencies,
                operator_max_latencies,
                edge_avg_latencies,
                edge_max_latencies,
                num_samples,
            )
        graph = create_graph(
            operator_avg_latencies,
            operator_max_latencies,
            edge_avg_latencies,
            edge_max_latencies,
            num_samples,
            highlight,
        )
        graph = add_graph_labels(graph, num_samples)
        graph.write(output_filename)
        if verbose:
            print(
                "The graph with performance numbers is written to file {}".format(output_filename)
            )
    else:
        # live mode
        directory = filenames[0]
        operator_avg_latencies = {}
        operator_max_latencies = {}
        edge_avg_latencies = {}
        edge_max_latencies = {}
        num_samples = 0
        read_files = {}
        while True:
            log_files = [f for f in os.listdir(directory) if f.endswith(".log")]
            if len(log_files) == 0:
                if verbose:
                    print(
                        "No log files (files ending with .log extension) is found in the folder {}.".format(
                            directory
                        )
                    )
                    print("sleeping for 1 seconds")
                time.sleep(1)
                continue
            # sort the files in ascending order of their creation time
            log_files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
            for filename in log_files:
                if filename not in read_files:
                    read_files[filename] = 0
                prev_num_samples = num_samples
                (
                    operator_avg_latencies,
                    operator_max_latencies,
                    edge_avg_latencies,
                    edge_max_latencies,
                    num_samples,
                    read_lines,
                ) = parse_log(
                    os.path.join(directory, filename),
                    read_files[filename],
                    operator_avg_latencies,
                    operator_max_latencies,
                    edge_avg_latencies,
                    edge_max_latencies,
                    num_samples,
                    skip_begin_messages=10,
                    discard_last_messages=0,
                    livefile=True,
                )
                read_files[filename] += read_lines
                if prev_num_samples != num_samples:
                    graph = create_graph(
                        operator_avg_latencies,
                        operator_max_latencies,
                        edge_avg_latencies,
                        edge_max_latencies,
                        num_samples,
                        highlight,
                    )
                    if verbose:
                        print(
                            "Read file {} - Line number: {}".format(filename, read_files[filename])
                        )
                    graph = add_graph_labels(graph, num_samples)
                    graph.write(output_filename)
                    if verbose:
                        print(
                            "The graph with performance numbers is updated in file {}".format(
                                output_filename
                            )
                        )

            if verbose:
                print("sleeping for 1 seconds")
            time.sleep(1)


if __name__ == "__main__":
    args = parse_arguments()
    create_dot_file(args.filenames, args.output, args.live, args.verbose, args.highlight)
