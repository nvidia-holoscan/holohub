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
import random

import matplotlib.pyplot as plt
import networkx as nx


def add_edges(graph, edgeset):
    for rawedge in edgeset.rstrip("\n").split(")"):
        if rawedge != "":
            processededge = rawedge.split(",")
            graph.add_edge(
                processededge[0].lstrip(("('")).rstrip("'"),
                processededge[1].lstrip(("( '")).rstrip("'"),
            )


def propose_execution_times(G, numvars):

    nodes = len(list(G.nodes))

    if numvars == 0:
        n = nodes
    else:
        n = numvars

    if nodes > 10:
        upper = 10
    else:
        upper = nodes

    for i in range(n):
        exec_times = random.sample(range(100, 100 * upper), nodes)
        max_value = max(exec_times)
        max_index = exec_times.index(max_value)

        for j in range(nodes):
            G.nodes[list(G.nodes)[j]]["executiontimes" + str(i)] = exec_times[j]

        if numvars == 0:
            # Maximize one node's waiting time, so we have one iteration where each node is the max
            temp = G.nodes[list(G.nodes)[i]]["executiontimes" + str(i)]
            G.nodes[list(G.nodes)[i]]["executiontimes" + str(i)] = max_value
            G.nodes[list(G.nodes)[max_index]]["executiontimes" + str(i)] = temp


def construct_graphs(source):
    linecounter = 1
    graphs = []

    for line in source:
        if linecounter % 3 == 1:
            unconditionalEdges = line
        if linecounter % 3 == 2:
            ifEdges = line
        if linecounter % 3 == 0:
            elseEdges = line

            areifEdges = True if ifEdges != "\n" else False
            areelseEdges = True if elseEdges != "\n" else False

            if areifEdges and areelseEdges:
                graph = nx.DiGraph()
                graphalt = nx.DiGraph()
                add_edges(graph, unconditionalEdges)
                add_edges(graphalt, unconditionalEdges)
                add_edges(graph, ifEdges)
                add_edges(graphalt, elseEdges)
                graphs.append(graph)
                graphs.append(graphalt)
            elif areifEdges:
                graph = nx.DiGraph()
                graphalt = nx.DiGraph()
                add_edges(graph, unconditionalEdges)
                add_edges(graphalt, unconditionalEdges)
                add_edges(graphalt, ifEdges)
                graphs.append(graph)
                graphs.append(graphalt)
            else:
                graph = nx.DiGraph()
                add_edges(graph, unconditionalEdges)
                graphs.append(graph)

        linecounter += 1
    return graphs


def visualize_graphs(graphs):
    index = 1
    for G in graphs:
        number = 0
        labels = {}
        for node in list(nx.topological_sort(G)):
            labels[node] = number
            number += 1
        # G = nx.convert_node_labels_to_integers(G, label_attribute="operator")
        # nx.draw(G, with_labels=True, font_weight='bold')
        nx.draw(
            G, labels=labels, pos=nx.bfs_layout(G, next(nx.topological_sort(G))), font_weight="bold"
        )

        plt.savefig("graph" + str(index) + ".png", dpi=1200)
        plt.clf()
        index += 1


def get_unique(graphs):
    unique = []
    for G in graphs:
        repeat = False
        for H in unique:
            if nx.is_isomorphic(G, H):
                repeat = True
        if nx.is_connected(G.to_undirected()) and not repeat:
            unique.append(G)
    return unique


def retstructure(graph, numvars):
    ret = []
    edgesret = []

    reference = dict(zip(list(nx.topological_sort(graph)), list(range(len(graph.nodes)))))

    for edge in graph.edges:
        edgesret.append((reference[edge[0]], reference[edge[1]]))

    for i in range(numvars):
        nodesret = []

        for node in reference.keys():
            val = graph.nodes[node]["executiontimes" + str(i)]
            nodesret.append((val, val))

        ret.append((nodesret, edgesret))

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    source = open(args.file, "r")

    graphs = construct_graphs(source)

    unique = get_unique(graphs)

    # nx.draw(unique[5], with_labels=True, pos=nx.planar_layout(unique[5]), font_weight='bold')# next(nx.topological_sort(unique[5]))), )

    # plt.savefig('wat.png', dpi = 1200)

    visualize_graphs(unique)

    for graph in unique:
        # predsucc.write(str(len(graph.nodes)) + "\n")

        for node in list(graph.nodes):
            pass
            # print(list(graph.predecessors(node)))
        #    predsucc.write(node + " predecessors: " + "\n")
        #    for edge in graph.predecessors(node):
        #        predsucc.write(edge + "\n")
        #    predsucc.write("\n")
        #    predsucc.write(node + " successors: " + "\n")
        #    for edge in graph.successors(node):
        #        predsucc.write(edge + "\n")
        #    predsucc.write("\n")
        # predsucc.write("\n")

    # print(len(unique))
