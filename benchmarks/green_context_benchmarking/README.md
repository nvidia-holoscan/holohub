# Green Context CUDA Kernel Launch-Start Time Benchmark

This benchmark measures CUDA kernel launch-start time improvements provided by NVIDIA CUDA Green Context technology in the Holoscan SDK framework using NVIDIA CUPTI (CUDA Profiling Tools Interface) for precise GPU timing measurements.

## Overview

The benchmark compares CUDA kernel launch-start times with and without Green Context by measuring the **actual time from kernel launch to GPU execution start** using NVIDIA CUPTI under realistic GPU contention scenarios. The benchmark uses a controlled A/B testing approach to isolate the pure Green Context benefit from general stream isolation effects.

### What it Measures

- **CUDA Kernel Launch-Start Time**: CUPTI-measured time from `cudaLaunchKernel()` call to actual GPU execution start
- **Background Load Performance**: DummyLoadOp execution timing statistics
- **Performance Improvement**: Launch-start time reduction with Green Context isolation
- **Tail Latency**: P95/P99 improvements for predictable real-time performance
- **Distribution Analysis**: How Green Context affects timing consistency

## Architecture

### Components

1. **TimingBenchmarkOp**: Measures CUPTI-based CUDA kernel launch-start time using lightweight kernels
2. **DummyLoadOp**: Creates realistic GPU contention using compute-intensive kernels
3. **CuptiSchedulingProfiler**: NVIDIA CUPTI-based profiler for accurate launch-start time measurement
4. **Green Context Setup**: Configures GPU partitioning for isolated execution


### A/B Testing Design

The benchmark uses a controlled comparison to isolate Green Context benefits:

**Baseline:** Both kernels run on separate non-default CUDA streams WITHOUT Green Context partitions
**Green Context:** Both kernels run on separate non-default CUDA streams WITH Green Context partitions

This design ensures that any performance difference is due to Green Context partitioning, not simply moving off the default stream.


## Usage

### Basic Usage

```bash
./holohub run green_context_benchmarking --docker-opts="--user root"
```

### Command Line Options

```bash
./holohub run green_context_benchmarking --docker-opts="--user root" --run-args="[OPTIONS]"

Options:
  --samples N           Number of timing samples to measure (default: 1000)
  --load-intensity N    GPU load intensity multiplier (default: 20)
  --workload-size N     GPU memory size in MB for DummyLoadOp (default: 8)
  --threads-per-block N CUDA threads per block for GPU kernels (default: 512)
  --mode MODE          Run mode: 'baseline', 'green-context', or 'all' (default: all)
                        baseline: Run only without green context
                        green-context: Run only with green context
                        all: Run both and show comparison
  --help               Show this help message

Examples:
  ./holohub run green_context_benchmarking --docker-opts="--user root" --run-args="--help"
  ./holohub run green_context_benchmarking --docker-opts="--user root" --run-args="--samples 1000 --load-intensity 10 --mode all"
  ./holohub run green_context_benchmarking --docker-opts="--user root" --run-args="--workload-size 16 --threads-per-block 256 --mode baseline"
```

### Parameters Explained

- **samples**: Number of launch-start time measurements per scenario (1000+ recommended for stable results)
- **load-intensity**: Computational intensity of background GPU workload (10-1000 range)
- **workload-size**: Memory footprint in MB for GPU kernels
- **threads-per-block**: CUDA thread block size for optimal GPU utilization
- **mode**: Which benchmark scenarios to run (baseline, green-context, or both)

## Sample Output

```
================================================================================
Green Context CUDA Kernel Start Time Benchmark
================================================================================
Benchmark Configurations:
  Benchmark Mode: all
  Measurement Samples: 1000
  Background Load Intensity: 20
  Background Load Size: 8 MB (2097152 elements)
  CUDA Threads Per Block: 512

Initializing CUPTI profiler...
[CUPTI] Successfully initialized scheduling latency profiler

================================================================================
Running benchmark for baseline
(non-default CUDA streams, without green context)
================================================================================
[info] [green_context_benchmark.cpp:302] [TimingBenchmarkOp] Collecting 1/1000 samples
[info] [green_context_benchmark.cpp:302] [TimingBenchmarkOp] Collecting 100/1000 samples
[info] [green_context_benchmark.cpp:302] [TimingBenchmarkOp] Collecting 200/1000 samples

...
Baseline benchmark completed

================================================================================
Running main benchmark
(with green context, separate partitions for each kernel)
================================================================================
[info] [green_context_benchmark.cpp:302] [TimingBenchmarkOp] Collecting 1/1000 samples
[info] [green_context_benchmark.cpp:302] [TimingBenchmarkOp] Collecting 100/1000 samples
...
Main benchmark completed

================================================================================
Benchmark Configurations
================================================================================
  Benchmark Mode: all
  Measurement Samples: 1000
  Background Load Intensity: 20
  Background Load Size: 8 MB (2097152 elements)
  CUDA Threads Per Block: 512

================================================================================
Benchmark Results
================================================================================
=== Without Green Context (Baseline) ===
CUDA Kernel Launch-Start Time:
  Average: 241.29 μs
  Std Dev: 168.54 μs
  Min:     2.31 μs
  P50:     252.35 μs
  P95:     465.74 μs
  P99:     472.84 μs
  Max:     744.14 μs
  Samples: 1000

=== With Green Context ===
CUDA Kernel Launch-Start Time:
  Average: 5.14 μs
  Std Dev: 2.77 μs
  Min:     2.78 μs
  P50:     4.07 μs
  P95:     10.42 μs
  P99:     15.37 μs
  Max:     24.85 μs
  Samples: 1000

================================================================================
Baseline and Green Context Benchmark Comparison
================================================================================
Launch-Start Latency:
  Average Latency:    241.29 μs →     5.14 μs  (+97.87%)
  95th Percentile:   +465.74 μs →   +10.42 μs  (+97.76%)
  99th Percentile:   +472.84 μs →   +15.37 μs  (+96.75%)

================================================================================
Dummy Load Execution Time Statistics
================================================================================
=== Without Green Context (Baseline) ===
  Average: 503.75 μs
  Std Dev: 18.99 μs
  Samples: 235707

=== With Green Context ===
  Average: 532.27 μs
  Std Dev: 17.78 μs
  Samples: 281813
```

## Understanding Results

### Key Metrics

- **CUPTI-based Launch-Start Time**: Hardware-measured time from kernel launch to GPU execution start
- **Average/P95/P99 Percentiles**: Statistical distribution of launch-start times
- **Background Load Statistics**: DummyLoadOp execution timing statistics

### What to Expect

**Typical Performance Patterns:**
- **Without Green Context**: Higher average launch-start times with significant variability and inconsistent performance
- **With Green Context**: Much lower and more consistent launch-start times with reduced variability
- **Performance Gains**: Substantial reduction in launch-start times across all percentiles (average, P95, P99)
- **Consistency Improvement**: Green Context typically shows much better timing consistency and predictability

**⚠️ Important Environmental Factor:**
**Display Connection Impact**: Having monitors connected via DisplayPort to the GPU significantly degrades launch-start time performance. For optimal benchmark results, disconnect displays and access the system via SSH/remote connection.

### Warning Messages

The benchmark may include CUPTI-related warnings:

```bash
[CUPTI] WARNING: Activity buffer may have overflowed
[CUPTI] Data polling timed out after 500 attempts
```

These indicate potential measurement issues under high GPU contention.

## Technical Details

### GPU Workload

**Timing Kernel (simple_benchmark_kernel):**
- Lightweight computation (sin/cos operations)
- Fixed elements (1024)
- 256 threads per block
- Designed for minimal execution time to isolate launch-start latency

**Background Load (background_load_kernel):**
- Heavy computational loops with transcendental functions
- Memory access patterns to stress memory subsystem
- Configurable intensity (`--load-intensity`) and workload size (`--workload-size`)
- Configurable threads per block (`--threads-per-block`)
- Runs on separate non-default stream to create realistic GPU contention
- In Green Context mode: runs in dedicated partition to test isolation

### Green Context Configuration

The benchmark dynamically calculates optimal partition sizes:

```cpp
// Dynamic partition sizing (roughly half total SMs per partition)
int total_sms = prop.multiProcessorCount;
int sms_per_partition = std::max(4, (total_sms / 2) & ~3);  // Multiple of 4 SMs

std::vector<uint32_t> partitions = {sms_per_partition, sms_per_partition};

// Separate partitions for each workload
// Partition 0: DummyLoadOp (background contention)
// Partition 1: TimingBenchmarkOp (latency measurement)
```

This ensures proper isolation with each workload getting dedicated GPU resources.

## Troubleshooting

### Common Issues

1. **Green Context not available**:
   - Check GPU compute capability: `deviceQuery`
   - Verify GPU has enough SMs: minimum 8 SMs required (4 per partition × 2 partitions)

2. **High variability**:
   - Increase `--samples` to 1000+ for more stable results
   - Check system load and background processes

3. **Low contention or insufficient GPU stress**:
   - Increase `--load-intensity` to create more background computation
   - Increase `--workload-size` to use more GPU memory for background load

### Performance Tuning

- **Low contention scenarios**: Increase `--load-intensity` to 20-100
- **More stable results**: Use `--samples 1000+` (default)
- **Large GPUs**: Increase `--workload-size` to 16-32MB
- **Memory-limited systems**: Reduce `--workload-size` to 4-8MB
- **Different thread configurations**: Adjust `--threads-per-block` (256, 512, 1024)
- **Specific testing**: Use `--mode baseline` or `--mode green-context` for focused testing

---
