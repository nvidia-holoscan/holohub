# Holoscan SDK Response-Time Analysis 

We have performed a theoretical response-time analysis of applications created using Holoscan SDK in an RTSS paper [1]. This work accounts for different queuing delays due to different types of connections and dependencies between the operators of a Holoscan application. This directory contains helpful scripts for the timing analysis of Holoscan applications, based on the paper.

Detailed instructions for how to reproduce the results of the paper, along with the code, can be found in the `artifact` directory.

## Scripts

The scripts in current directory (`computeWCRT.py` and `runsimulation.py`) are written in Python and require the `networkx` and `pydot` packages, which can be easily installed with pip.

    pip install networkx
    pip install pydot

Each script takes a representation of an application graph in the form of a DOT file. An example DOT file is provided in `exampledot.dot`. The DOT file defines a Holoscan application graph in two parts. The first section consists of the names of the operators, with their worst-case execution times as attributes. The second section captures the connections between operators.

We assume that there is only one root operator and one leaf operator in the application graph, without losing any generality. More information on why this works for more than one root and leaf can be found in the paper [1].


### `computeWCRT.py`

This script takes a path to a DOT file, representing a Holoscan application graph and computes an upper bound of worst-case response time (WCRT) for the application. The script does not run a Holoscan application on actual hardware. Instead, it computes the WCRT following the timing analysis done in our paper [1]. The algorithm runs reasonably fast (within quadratic time) for graphs with up to 20-30 nodes.

    python computeWCRT.py examplegraph.dot
    Worst-case response time: 1600

This script can also, optionally, account for extra scheduling overheads, empirically observed on Jetson AGX Orin. The `-overhead` argument can include this overhead in the WCRT analysis. Kindly note that this is a heuristic based on measurements using a Jetson AGX Orin and will not be accurate for all systems.

    python computeWCRT.py examplegraph.dot --overhead
    Worst-case response time: 1612

### `runsimulation.py` 

This script takes a path to a DOT file representing a Holoscan application graph, an expected runtime, and a root operator period (in this order), and runs a discrete-event simulation of the execution of the Holoscan application under the given conditions, printing the results. The runtime argument determines how long the simulation will run. For example, if an application takes 1000 time units to process an input, and the runtime is 1100 time units, then only one iteration will be simulated. The period argument determines how often the source operator can execute. If the source operator could execute every 50 time units, but period is 100 time units, then the source will be constrained in this script. 

The period will affect response times, though not necessarily the worst-case response time. For example, lowering the period may increase the queuing time of early iterations with response times still converging to the same value. In the 
output below, changing the period to `5` would increase the response time of iteration 1 by 5, but leave the WCRT unchanged.

    python runsimulation.py examplegraph.dot 3000 10
    Iteration 0
    Source start:  0
    Sink finish:   900
    Response time: 900

    Iteration 1
    Source start:  10
    Sink finish:   1300
    Response time: 1290

    Iteration 2
    Source start:  200
    Sink finish:   1700
    Response time: 1500

    Iteration 3
    Source start:  500
    Sink finish:   2100
    Response time: 1600

    Iteration 4
    Source start:  900
    Sink finish:   2500
    Response time: 1600

    Iteration 5
    Source start:  1300
    Sink finish:   2900
    Response time: 1600

    Worst-case response time: 1600


For this application, response times converge to the worst-case response time of 1600. Note that not all applications will converge to single value in this manner, and response times may increase or decrease periodically even after reaching the worst-case response time. The previously mentioned `computeWCRT.py` script provides a theoretical upper bound, even though it can be more pessimistic than simulated or real-world observed times. More details are available in our paper [1].

### Citation

[1] P. Schowitz, S. Sinha, and A. Gujarati, “Response-Time Analysis 
of a Soft Real-time NVIDIA Holoscan Application,” in IEEE Real-Time 
Systems Symposium, 2024.

BibTeX:

    @inproceedings{Schowitz2024,
    author    = {P. Schowitz and S. Sinha and A. Gujarati},
    title     = {Response-Time Analysis of a Soft Real-time NVIDIA Holoscan Application},
    booktitle = {Proceedings of the IEEE Real-Time Systems Symposium},
    year      = {2024},
    }