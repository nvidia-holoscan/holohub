# SPDX-FileCopyrightText: Copyright (c) 2025 UNIVERSITY OF BRITISH COLUMBIA. All rights reserved.
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

import networkx as nx


def get_response_time(DG, source, sink, overhead):

    # Find immediate postdominators for every node by finding dominators of the edge reversed graph
    postdominators = nx.algorithms.immediate_dominators(DG.reverse(), sink)

    waitingtimes = {}

    # For every node in the graph, find its worst-case inter-processing delay
    for node in DG.nodes:
        if DG.out_degree(node) > 1:
            maxcost = 0
            paths = nx.all_simple_paths(DG, node, postdominators[node])
            for path in paths:
                pathcost = nx.path_weight(DG, path, "weight")
                if maxcost < pathcost:
                    maxcost = pathcost
            waitingtimes[node] = maxcost
        else:
            waitingtimes[node] = DG.nodes[node]["WCET"]

    WCRTcandidates = {}

    # Find the path from the source to the sink with the greatest sum of execution times
    maxcost = 0
    paths = nx.all_simple_paths(DG, source, sink)
    for path in paths:
        pathcost = nx.path_weight(DG, path, "weight")
        if maxcost < pathcost:
            maxcost = pathcost
            longestpath = path

    longestpathcost = maxcost + waitingtimes[sink]

    for node in DG.nodes:

        shortestsourcetonodepath = nx.shortest_path(DG, source, node)

        sourcetobottle = (len(shortestsourcetonodepath)) * waitingtimes[node]

        # Find the path from the node to the sink with the greatest sum of execution times
        maxcost = 0
        paths = nx.all_simple_paths(DG, node, sink)
        for path in paths:
            pathcost = nx.path_weight(DG, path, "weight")
            if maxcost < pathcost:
                maxcost = pathcost

        # Add the execution time of the sink operator, since it is encoded as an edge weight
        bottletosink = maxcost + waitingtimes[sink]

        WCRTcandidates[node] = sourcetobottle + bottletosink

        if node not in longestpath:

            WCRTcandidates[node] += longestpathcost - nx.path_weight(
                DG, shortestsourcetonodepath, "weight"
            )

    WCRT = max(WCRTcandidates.values())

    n = len(DG.nodes)

    if overhead:
        return WCRT + (n * 3)
    else:
        return WCRT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--overhead", action="store_true")
    args = parser.parse_args()

    DG = nx.DiGraph(nx.nx_pydot.read_dot(args.file))

    source = [node for node in DG.nodes if DG.in_degree(node) == 0]

    sink = [node for node in DG.nodes if DG.out_degree(node) == 0]

    for node, data in DG.nodes(data=True):
        data["WCET"] = int(data["WCET"])

    for u, v, data in DG.edges(data=True):
        data["weight"] = DG.nodes(data=True)[u]["WCET"]

    print(
        "Worst-case response time:", str(get_response_time(DG, source[0], sink[0], args.overhead))
    )
