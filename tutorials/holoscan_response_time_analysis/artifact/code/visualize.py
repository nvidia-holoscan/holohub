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

import matplotlib.pyplot as plt
import numpy as np

colors = ["brown", "green", "purple", "red", "olive", "darkslategray", "deeppink", "blue"]
markers = ["x", "o", "^", "s", "D", "P", "*", "1"]
labels = ["A", "B", "C", "D", "E", "F", "G", "H"]


def main(graphs, constraint=False):
    graphobv = plt.figure(0)
    graphsim = plt.figure(1)

    if constraint:
        predicted = open("constraintresponsetimes.txt", "r")
    else:
        predicted = open("predictedresponsetimes.txt", "r")
    observed = open("observedresponsetimes.txt", "r")
    simul = open("simulatedresponsetimes.txt", "r")

    for i, data in enumerate(graphs[0]):
        predictedvals = []
        observedvals = []
        simulvals = []

        for exp in range(data[0]):
            predictedvals.append(int(predicted.readline().split(" ")[-1].rstrip()))
            observedvals.append(int(float(observed.readline().split(" ")[-1].rstrip())))
            simulvals.append(int(simul.readline().split(" ")[-1].rstrip()))

        plt.figure(0)
        plt.scatter(
            observedvals, predictedvals, color=colors[i], marker=markers[i], label=labels[i], s=30
        )
        plt.figure(1)
        plt.scatter(
            simulvals, predictedvals, color=colors[i], marker=markers[i], label=labels[i], s=30
        )

    plt.figure(0)
    plt.plot([1000, 8500], [1000, 8500], color="black", linestyle="--")
    plt.ylabel("Predicted WCRT (ms)", fontsize=20)
    plt.xlabel("Observed WCRT (ms)", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=15, ncol=2)

    graphobv.savefig("evalpessimism.pdf", bbox_inches="tight")

    plt.figure(1)
    plt.plot([500, 8500], [500, 8500], color="black", linestyle="--")
    plt.ylabel("Predicted WCRT (ms)", fontsize=20)
    plt.xlabel("Simulated WCRT (ms)", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=15, ncol=2)
    graphsim.savefig("evalsim.pdf", bbox_inches="tight")


def overheadmain():
    graph = plt.figure(2)

    base = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])

    observed = open("observedoverheads.txt", "r")

    observedoverheads = [int(float(x.split(" ")[-1].rstrip())) for x in observed.readlines()]

    plt.ylabel("Response time (ms)", fontsize=20)
    plt.xlabel("# Operators", fontsize=20)

    plt.xticks(np.arange(1, 12, 1))

    plt.plot(range(1, 12), base, color="blue", marker="o", label="Theoretical WCRT")
    plt.plot(
        range(1, 12),
        observedoverheads,
        color="red",
        marker="o",
        linestyle="--",
        label="Observed WCRT",
    )
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    graph.savefig("evaloverhead.pdf", bbox_inches="tight")


def scalabilitymain(graphs, constraint=False):

    labels = ["20 edges", "40 edges", "60 edges", "80 edges", "100 edges"]

    translate = dict(
        [(19, 0), (20, 0), (39, 1), (40, 1), (59, 2), (60, 2), (79, 3), (80, 3), (99, 4), (100, 4)]
    )

    graphsim = plt.figure(3)

    if constraint:
        predicted = open("constraintresponsetimes.txt", "r")
    else:
        predicted = open("predictedresponsetimes.txt", "r")
    simul = open("simulatedresponsetimes.txt", "r")

    for nodecount, edgecount in graphs[0]:
        predictedvals = []
        simulvals = []

        i = translate[edgecount]

        numvars = 10

        for exp in range(numvars):
            predictedvals.append(int(predicted.readline().split(" ")[-1].rstrip()))
            simulvals.append(int(simul.readline().split(" ")[-1].rstrip()))

        plt.figure(3)
        plt.scatter(
            simulvals, predictedvals, color=colors[i], marker=markers[i], label=labels[i], s=30
        )

    plt.figure(3)
    plt.plot([1000, 40000], [1000, 40000], color="black", linestyle="--")
    plt.ylabel("Predicted WCRT (ms)", fontsize=20)
    plt.xlabel("Simulated WCRT (ms)", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=13)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=13)

    graphsim.savefig("evalscalability.pdf", bbox_inches="tight")

    predicted.close
    simul.close


def scalabilitypess():
    translate = dict(
        [(19, 0), (20, 0), (39, 1), (40, 1), (59, 2), (60, 2), (79, 3), (80, 3), (99, 4), (100, 4)]
    )
    data = [[], [], [], [], []]

    with open("predictedresponsetimes.txt", "r") as predicted, open(
        "simulatedresponsetimes.txt", "r"
    ) as simul:
        for line1, line2 in zip(predicted, simul):

            edges = int(line2.strip().split()[0])
            predicted = float(line1.strip().split()[-1])
            real = float(line2.strip().split()[-1])

            data[translate[edges]].append((predicted - real) / predicted)

    graphscaled = plt.figure(5)

    # Combine data into a list of datasets

    # Create the box plots
    plt.boxplot(data, vert=True, patch_artist=True)

    # Add titles and labels
    plt.xlabel("Edge Count", fontsize=20)
    plt.ylabel("Relative Pessimism", fontsize=20)

    # Customize the x-axis tick labels
    plt.xticks([1, 2, 3, 4, 5], ["20", "40", "60", "80", "100"], fontsize=15)
    plt.yticks(fontsize=15)

    graphscaled.savefig("evalscalabilitypess.pdf", bbox_inches="tight")


def timinganalysis():
    x = []
    y = []
    with open("analysistimes.txt", "r") as timing:
        for line in timing:
            x.append(int(line.split(" ")[0]))
            y.append(float(line.split(" ")[1].rstrip()))

    graphanalysis = plt.figure(6)

    # Create the scatter plot
    plt.scatter(x, y, color="black", marker="x", label="A", s=20)

    plt.yscale("log")
    plt.ylim(1e-4, 1e1)

    # Add labels
    plt.ylabel("Analysis Time\n(seconds, log scale, base 10)", fontsize=20, multialignment="center")

    plt.xlabel("Node Count", fontsize=20)
    plt.xticks(fontsize=15)  # Increase x-axis tick labels font size
    plt.yticks(fontsize=15)  # Increase y-axis tick labels font size

    graphanalysis.savefig("evalanalysis.pdf", bbox_inches="tight")
