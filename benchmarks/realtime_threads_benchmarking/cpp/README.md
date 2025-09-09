# C++ Real-time Thread Benchmark

This directory contains the C++ implementation of the real-time thread benchmarking application for Holoscan.

## Overview

The C++ benchmark provides the same functionality as the Python version but with improved performance and lower overhead. It measures the effectiveness of real-time thread scheduling policies in Holoscan applications.

## Building

The benchmark requires the following dependencies:
- Holoscan SDK
- yaml-cpp
- CMake 3.16+

To build the benchmark:

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Running the Benchmark

```bash
./realtime_thread_benchmark [options]
```

### Options

- `--target-fps <fps>`: Target FPS (30 or 60, default: 60)
- `--duration <seconds>`: Benchmark duration in seconds (default: 30)
- `--scheduling-policy <policy>`: SCHED_DEADLINE, SCHED_FIFO, or SCHED_RR (default: SCHED_DEADLINE)
- `--load-duration-ms <ms>`: CPU work duration per load operator call (default: 20.0)
- `--output <file>`: Output YAML file (default: benchmark_results.yaml)
- `--help`: Show help message

### Example

```bash
# Run a 30-second benchmark at 60 FPS with SCHED_DEADLINE
./realtime_thread_benchmark --target-fps 60 --duration 30 --scheduling-policy SCHED_DEADLINE

# Run with custom load duration and output file
./realtime_thread_benchmark --load-duration-ms 15.0 --output my_results.yaml
```

## Output

The benchmark generates a JSON file containing detailed timing statistics for both normal and real-time scheduling modes. The output includes:

- Frame period statistics (mean, std dev, min, max)
- Execution time statistics (mean, std dev, min, max)
- Raw timing data for post-processing
- Configuration parameters

## Post-processing with Python

Use the `plot.py` script to generate visualizations from the JSON results:

```bash
python3 plot.py --input benchmark_results.json --output-dir ./plots
```

This will generate:
- `timing_over_time.png`: Frame periods and execution times over time
- `simple_histograms.png`: Distribution histograms

## Architecture

The benchmark consists of several operators:

1. **TargetOperator**: Measures timing performance and aims to run at a specific FPS
2. **LoadOperator**: Consumes CPU resources to create contention
3. **DataSinkOperator**: Receives data from other operators

The application creates separate thread pools for real-time and normal scheduling, allowing direct comparison of timing consistency.

## Performance Considerations

- The C++ version has lower overhead than the Python version
- Uses high-resolution timers for precise measurements
- Minimal memory allocation during timing measurements
- Thread-safe statistics collection

## Troubleshooting

- Ensure you have sufficient permissions for real-time scheduling (may require sudo)
- Check that yaml-cpp is properly installed
- Verify Holoscan SDK installation and environment setup
