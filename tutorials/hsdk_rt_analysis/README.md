# Holoscan SDK Response-Time Analysis 

This directory contains helpful scripts for timing analysis of Holoscan apps, based on published research regarding the Holoscan SDK (see citation below).

Detailed instructions for how to reproduce the results of the paper, along with the code, can be found in the `artifact` directory.

## Scripts

The scripts in this directory are written in Python and require the `networkx` and `pydot` packages, which can be easily installed with pip. Each script takes a representation of an application graph in the form of a DOT file. An example DOT file, `exampledot.dot`, can be used as a model. The graph defined in the DOT file is a simplified view of a Holoscan application, with two parts. The first section consists of the name of each operator, with its worst-case execution time as an attribute. The second section captures data flow between operators. We assume that there is exactly one operator with no inputs, and exactly one operator with no outputs.   


### computeWCRT.py

This script takes a path to a DOT file and computes a theoretical worst-case response bound for the application.

    python computeWCRT.py examplegraph.dot
    Worst-case response time: 1600

Optionally, the overhead argument adds some extra pessimism to simulate various overheads within Holoscan that are not
captured as part of the worst-case execution time of an operator. Note that this is a heuristic based on measurements using a Jetson AGX Orin and will not be accurate for all systems.

    python computeWCRT.py examplegraph.dot --overhead
    Worst-case response time: 1612

### runsimulation.py

This script takes a path to a DOT file, a runtime, and a period (in this order), and runs a discrete-event simulation of the execution of the Holoscan application under the given conditions, printing the results. The runtime argument determines how long the simulation will run. For example, if an application takes 1000 units to process an input, and the runtime is 1100, then only one iteration will be simulated. The period argument determines how often the source operator can execute. If the source operator could execute every 50 units, but period is 100, then the source will be constrained. The period will affect response times, though not necessarily the worst-case response time. 

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

For this application, response times converge to the worst-case response time of 1600. Note that not all applications will converge to single value in this manner, and response times may increase or decrease periodically.

### Citation

P. Schowitz, S. Sinha, and A. Gujarati, “Response-Time Analysis 
of a Soft Real-time NVIDIA Holoscan Application,” in IEEE Real-Time 
Systems Symposium, 2024.

BibTeX:

    @inproceedings{Schowitz2024,
    author    = {P. Schowitz and S. Sinha and A. Gujarati},
    title     = {Response-Time Analysis of a Soft Real-time NVIDIA Holoscan Application},
    booktitle = {Proceedings of the IEEE Real-Time Systems Symposium},
    year      = {2024},
    }