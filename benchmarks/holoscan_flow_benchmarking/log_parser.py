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


def parse_line_from_log(line):
    operators = line.split("->")
    # print (operators)
    op_timestamps = []
    for operator in operators:
        # trim whitespaces for left and right side
        # print ("op: ", operator)
        op_name_timestamp = operator.strip().rstrip()[1:-1]
        op_timestamps.append(op_name_timestamp.split(","))
    return op_timestamps


# return a path and latency pair where a path is string of operators separated by path_separator
def get_path_latency(op_timestamps, path_separator="→ "):
    operators = []
    latency = 0
    for op_timestamp in op_timestamps:
        operators.append(op_timestamp[0])
    # convert the latency to ms
    latency = float(int(op_timestamps[-1][2]) - int(op_timestamps[0][1])) / 1000.0
    path = path_separator.join(operators)
    return path, latency


# parse the log file and return all the latencies for each path
# The format is (Operator1, receive timestamp, publish timestsamp) -> (Operator2, receive timestamp,
# publish timestsamp) -> ... -> (OperatorN, receive timestamp, publish timestsamp)
def parse_log_as_paths_latencies(log_file):
    with open(log_file, "r") as f:
        paths_latencies = {}
        for line in f:
            # print ("line: ", line)
            if line[0] == "(":
                path_latency = get_path_latency(parse_line_from_log(line))
                if path_latency[0] in paths_latencies:
                    paths_latencies[path_latency[0]].append(path_latency[1])
                else:
                    paths_latencies[path_latency[0]] = [path_latency[1]]
        return paths_latencies
