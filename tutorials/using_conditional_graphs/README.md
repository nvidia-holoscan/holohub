# Using a Conditional Gate

![image](graph_with_condition_gate.png)

## Overview

Use a condition in an operator as a gate for a downstream operator

## Description
Basic example with tx and rx operators. Once a condition is met, 
the mx operator is used to multiply the value from the tx operator, before it is received by a new rx operator.

A conditional gate can be used to control the flow of data to a downstream branch of the directed acyclic branch. An example use case is a phase detection model, that triggers the execution of an inference operator for this particular phase. 

## Tutorial Instructions


```bash
python tutorials/using_conditional_graphs/conditional_execution_app.py
```

## Profiling
The graph image at the top of the page is created, using the [flow benchmarking tools](../../benchmarks/holoscan_flow_benchmarking/README.md) of holohub.
First run the patched application:
```bash
python benchmarks/holoscan_flow_benchmarking/benchmark.py --run-command " python tutorials/using_conditional_graphs/conditional_execution_app.py" --sched greedy
```
Then create a graph.dot file based on the logged data.
```bash
python benchmarks/holoscan_flow_benchmarking/app_perf_graph.py -o log_directory_20240513-121809/graph.dot log_directory_20240513-121809/logger_greedy_1_1.log 
```
Visualize the graph using xdot
```bash
xdot log_directory_20240513-121809/live_app_graph.dot
```


