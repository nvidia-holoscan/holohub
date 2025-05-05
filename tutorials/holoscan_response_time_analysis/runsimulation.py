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
import heapq

import networkx as nx

queuesize = 1


class operator(object):
    def __init__(self, name, predecessors, successors, WCET, sink):
        self.name = name
        self.predecessors = predecessors
        self.successors = successors
        self.WCET = WCET
        self.sink = sink
        self.working = False
        self.queue = dict(zip(predecessors, [0] * len(predecessors)))
        self.starttimes = []
        self.finishtimes = []


# Pushes the completion event for an operator's execution onto the heap, sets it to working, and removes the first item from its queue
def add_operator_exec(operator, eventqueue, time):
    heapq.heappush(eventqueue, (time + operator.WCET, operator.name))
    operator.working = True
    for port in operator.queue:
        operator.queue[port] -= 1


# Checks if an operator can execute, based on whether it is working and its queue + successor queues (downstream)
def can_op_execute(tag, operators):
    if tag == "periodic":
        return False

    operator = operators[tag]

    if operator.working == True:
        return False

    # Own queue must be full and corresponding output queues must be empty
    if 0 not in operator.queue.values():
        if operator.sink:
            return True
        for successor in operator.successors:
            if operators[successor].queue[operator.name] == queuesize:
                return False
        return True


# Recursively checks whether predecessors can now execute
def recursive_predecessor_check(operator, eventqueue, time, source, operators):
    if operator == source:
        return
    for predecessor in operator.predecessors:
        if can_op_execute(predecessor, operators):
            add_operator_exec(operators[predecessor], eventqueue, time)
            recursive_predecessor_check(operators[predecessor], eventqueue, time, source, operators)


# Given a tag corresponding to an operator that has completed execution, checks to see what new completion events should be added to the heap
def check_new_event(tag, eventqueue, time, period, source, operators):
    if tag == "periodic":

        interarrival = period

        heapq.heappush(eventqueue, (time + interarrival, "periodic"))
        # Add item to source's queue if empty
        if source.queue["periodic"] < queuesize:
            source.queue["periodic"] += 1
            # New item begins being processed by the DAG
            source.starttimes.append(time)
        # Begin execution of source if possible
        if can_op_execute(source.name, operators):
            add_operator_exec(source, eventqueue, time)

    else:
        for operator in operators.values():
            if operator.name == tag:
                operator.working = False
                operator.finishtimes.append(time)

                # First check if successors can execute given the new output from the operator
                if operator.successors is not None:
                    for successor in operator.successors:

                        # Check if there was queueing already in any queue of the successor operator
                        queueing = False
                        for queue in operators[successor].queue.values():
                            if queue != 0:
                                queueing = True
                        if not queueing:
                            operators[successor].starttimes.append(time)

                        # Add output to successor queues
                        operators[successor].queue[operator.name] += 1

                    for successor in operator.successors:
                        if can_op_execute(successor, operators):
                            add_operator_exec(operators[successor], eventqueue, time)
                            # Recursively check predecessors of successors in case of different paths to successor
                            recursive_predecessor_check(
                                operators[successor], eventqueue, time, source, operators
                            )

                # Next check if the operator can execute again
                if can_op_execute(operator.name, operators):
                    add_operator_exec(operator, eventqueue, time)

                # Finally, recursively check predecessors
                recursive_predecessor_check(operator, eventqueue, time, source, operators)


def run(stoptime, period, source, operators):
    clock = 0

    events = []

    # First item enters DAG
    check_new_event("periodic", events, clock, period, source, operators)

    while clock < stoptime:
        popped = heapq.heappop(events)
        clock = popped[0]

        check_new_event(popped[1], events, clock, period, source, operators)

    # Clean up for next run
    for operator in operators.values():
        operator.working = False
        operator.queue = dict.fromkeys(operator.queue, 0)


def runwithfile(graph, runtime, period):
    operators = {}

    for node in graph.nodes:
        pred = list(graph.predecessors(node))
        succ = list(graph.successors(node))
        if pred == []:
            operators[node] = operator(
                node, ["periodic"], succ, DG.nodes(data=True)[node]["WCET"], False
            )
            # If an operator has no precedecessors, it is source
            source = operators[node]
        elif succ == []:
            operators[node] = operator(node, pred, None, DG.nodes(data=True)[node]["WCET"], True)
            sink = operators[node]
        else:
            operators[node] = operator(node, pred, succ, DG.nodes(data=True)[node]["WCET"], False)

    run(runtime, period, source, operators)

    itemresponsetimes = []

    for i in range(len(sink.finishtimes)):
        print("Iteration", i)
        print("Source start: ", source.starttimes[i])
        print("Sink finish:  ", sink.finishtimes[i])
        itemresponsetimes.append(sink.finishtimes[i] - source.starttimes[i])
        print("Response time:", itemresponsetimes[i])
        print()

    print("Worst-case response time:", str(max(itemresponsetimes)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("runtime")
    parser.add_argument("period")
    args = parser.parse_args()

    DG = nx.DiGraph(nx.nx_pydot.read_dot(args.file))

    source = [node for node in DG.nodes if DG.in_degree(node) == 0]

    sink = [node for node in DG.nodes if DG.out_degree(node) == 0]

    for node, data in DG.nodes(data=True):
        data["WCET"] = int(data["WCET"])

    for u, v, data in DG.edges(data=True):
        data["weight"] = DG.nodes(data=True)[u]["WCET"]

    runwithfile(DG, int(args.runtime), int(args.period))
