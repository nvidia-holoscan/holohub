# Using a Conditional Gate

![image](graph_with_condition_gate.png)

## Overview

Use a condition in an operator as a gate for a downstream operator

## Description
Basic example with tx and rx operators. Once a condition is met, 
the mx operator is used to multiply the value from the tx operator, before it is received by a new rx operator.
## Tutorial Instructions


```bash
python tutorials/using_conditional_graphs/conditional_execution_app.py
```

## Profiling
```bash
python benchmarks/holoscan_flow_benchmarking/benchmark.py --run-command " python tutorials/using_conditional_graphs/conditional_execution_app.py" --sched greedy
```
```bash
python benchmarks/holoscan_flow_benchmarking/app_perf_graph.py -o log_directory_20240513-121809/graph.dot log_directory_20240513-121809/logger_greedy_1_1.log 
```
```bash
xdot log_directory_20240513-121809/live_app_graph.dot
```


