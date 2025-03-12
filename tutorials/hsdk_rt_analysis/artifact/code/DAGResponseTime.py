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

import timeit
from code.processDAGs import construct_graphs, get_unique, propose_execution_times

import networkx as nx


def get_response_time(DG, source, sink):

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
            # An edge case can occur in large graphs where the longest path through the graph does not
            # determine the response time.
            # This is described in the paper in more detail
            # For most graphs (including all practical examples) this below calculation results in
            # significant execution time increases for no decrease in pessimism, so we disable it
            # Note that the analysis is safe either way

            if False:
                # Find the path from the source to the node with the greatest sum of execution times
                maxcost = 0
                paths = nx.all_simple_paths(DG, source, node)
                for path in paths:
                    pathcost = nx.path_weight(DG, path, "weight")
                    if maxcost < pathcost:
                        maxcost = pathcost

                # greatestcostsourcetonodepathcost = maxcost

                # bottleneckpathcost = greatestcostsourcetonodepathcost + bottletosink

            WCRTcandidates[node] += longestpathcost - nx.path_weight(
                DG, shortestsourcetonodepath, "weight"
            )

    WCRT = max(WCRTcandidates.values())

    n = len(DG.nodes)

    overhead = n * 3

    return WCRT + overhead


def main(filepath, numvars, timing=False):
    source = open(filepath, "r")
    # For if you want to run with a saved set of execution times
    saved = None  # open("savedgeneratedexectimes.txt", "r")
    generatedexectimes = open("generatedexectimes.txt", "w")
    predictedresponsetimes = open("predictedresponsetimes.txt", "w")

    if timing:
        analysistimes = open("analysistimes.txt", "w")

    graphs = construct_graphs(source)

    unique = get_unique(graphs)

    graphindex = 1
    for graph in unique:

        if numvars == 0:
            n = len(graph.nodes)
        else:
            n = numvars

        propose_execution_times(graph, n)

        sorted = nx.topological_sort(graph)
        source = next(sorted)
        *_, sink = sorted

        generatedexectimes.write("Graph " + str(graphindex) + "\n")
        for i in range(n):
            for node in graph.nodes:
                if saved is None:
                    graph.nodes[node]["WCET"] = graph.nodes[node]["executiontimes" + str(i)]
                else:
                    saved.readline()
                    graph.nodes[node]["WCET"] = int(saved.readline().lstrip("   WCET: "))
                generatedexectimes.write(node + ": " + "\n")
                generatedexectimes.write(
                    "  WCET: " + str(graph.nodes[node]["executiontimes" + str(i)]) + "\n"
                )
                for send, to in graph.edges(node):
                    graph.edges[send, to]["weight"] = graph.nodes[node]["WCET"]

            if saved is not None:
                saved.readline()

            if timing:
                global testgraph
                testgraph = graph

                global testsource
                testsource = source

                global testsink
                testsink = sink

                test = (
                    timeit.timeit(
                        "get_response_time(testgraph, testsource, testsink)",
                        number=10,
                        globals=globals(),
                    )
                    / 10
                )
                analysistimes.write(str(len(graph.nodes)) + " " + str(test) + "\n")

            rt = get_response_time(graph, source, sink)

            predictedresponsetimes.write(
                "Graph "
                + str(graphindex)
                + " variation "
                + str(i)
                + " predicted WCET = "
                + str(rt)
                + "\n"
            )
            generatedexectimes.write("\n")
        graphindex += 1

    return [[(len(graph.nodes), len(graph.edges)) for graph in unique], unique]


if __name__ == "__main__":
    main("backgroundexps.txt", 10, True)
