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

**Default (older GPUs and Orin systems):**
```bash
./holohub run green_context_benchmarking --docker-opts="--user root"
```

**Modern GPUs and Jetson Thor:**
```bash
./holohub run green_context_benchmarking --docker-opts="--user root" --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu
```

### GPU Compatibility

**Use default command for:**
- Older GPUs (compute capability < 7.0, e.g., GTX 1080, GTX 1060, etc.)
- NVIDIA Jetson Orin systems

**Use v3.6.1-cuda13 image for:**
- Modern GPUs (compute capability ≥ 7.0, e.g., RTX 2080, RTX 3080, RTX 4090, RTX 6000 Ada Generation)
- NVIDIA Jetson Thor systems

**Note**: CUDA 13.0 dropped support for older GPU architectures (compute capability < 7.0). If you encounter `nvcc fatal : Unsupported gpu architecture 'compute_XX'` errors, use the default command instead.

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
  # Default (older GPUs and Orin)
  ./holohub run green_context_benchmarking --docker-opts="--user root" --run-args="--help"
  ./holohub run green_context_benchmarking --docker-opts="--user root" --run-args="--samples 1000 --load-intensity 10 --mode all"
  ./holohub run green_context_benchmarking --docker-opts="--user root" --run-args="--workload-size 16 --threads-per-block 256 --mode baseline"

  # Modern GPUs and Jetson Thor
  ./holohub run green_context_benchmarking --docker-opts="--user root" --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu --run-args="--samples 1000 --load-intensity 10"
  ./holohub run green_context_benchmarking --docker-opts="--user root" --base-img=nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu --run-args="--workload-size 16 --mode all"
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
Comprehensive Timing Results
================================================================================
=== Without Green Context (Baseline) ===
CUDA Kernel Launch-Start Time:
  Average: 249.09 μs
  Std Dev: 157.42 μs
  Min:     1.53 μs
  P50:     271.73 μs
  P95:     460.79 μs
  P99:     478.52 μs
  Max:     586.59 μs
  Samples: 1000

CUDA Kernel Execution Time:
  Average: 12.48 μs
  Std Dev: 3.98 μs
  Min:     1.34 μs
  P50:     13.44 μs
  P95:     16.83 μs
  P99:     17.54 μs
  Max:     18.59 μs
  Samples: 1000

=== With Green Context ===
CUDA Kernel Launch-Start Time:
  Average: 4.64 μs
  Std Dev: 2.37 μs
  Min:     2.78 μs
  P50:     3.88 μs
  P95:     10.04 μs
  P99:     14.01 μs
  Max:     23.98 μs
  Samples: 1000

CUDA Kernel Execution Time:
  Average: 1.18 μs
  Std Dev: 0.08 μs
  Min:     0.99 μs
  P50:     1.18 μs
  P95:     1.31 μs
  P99:     1.38 μs
  Max:     1.95 μs
  Samples: 1000

================================================================================
Baseline and Green Context Benchmark Comparison
================================================================================
Launch-Start Latency:
  Average Latency:    249.09 μs →     4.64 μs  (+98.14%)
  95th Percentile:   +460.79 μs →   +10.04 μs  (+97.82%)
  99th Percentile:   +478.52 μs →   +14.01 μs  (+97.07%)

Kernel Execution Time:
  Average Duration:    12.48 μs →     1.18 μs  (+90.57%)
  95th Percentile:    +16.83 μs →    +1.31 μs  (+92.21%)
  99th Percentile:    +17.54 μs →    +1.38 μs  (+92.16%)

================================================================================
Dummy Load Execution Time Statistics
================================================================================
=== Without Green Context (Baseline) ===
  Average: 513.20 μs
  Std Dev: 17.45 μs
  Samples: 262730

=== With Green Context ===
  Average: 532.74 μs
  Std Dev: 13.97 μs
  Samples: 289987
```

## Benchmark Results

### Executive Summary

Green Context delivers consistent, substantial performance improvements across both edge and high-end hardware:

| Platform | Best Case Improvement | Optimal Configuration | Launch-Start Time with GC |
|----------|----------------------|----------------------|------------|
| **Orin (16 SMs)** | 95.5% latency reduction | 4MB workload, 128-256 threads | 23-35μs |
| **RTX 6000 Ada (142 SMs)** | 97.9% latency reduction | 8-16MB workload, any thread count | 4-6μs |

### Detailed Results by Platform

**Performance Matrix Parameter Mapping**:
- **Load Int** → `--load-intensity`
- **Size (MB)** → `--workload-size`
- **Threads/Block** → `--threads-per-block`

#### Orin (16 SMs)

**Performance Matrix** :
```
Load  Size  Threads   Avg%    Baseline    GC
Int   (MB)  /Block    Impr    Avg(μs)     Avg(μs)
----------------------------------------------------
5     1     64        75.5    106.54      26.13
5     1     128       72.5    99.74       27.45
5     1     256       73.1    108.09      29.08
5     1     512       70.1    91.63       27.36
5     2     64        87.6    251.08      31.07
5     2     128       89.0    253.93      27.82
5     2     256       87.5    244.75      30.73
5     2     512       87.2    250.33      32.00
5     4     64        93.4    543.96      35.73
5     4     128       95.5    531.56      23.86
5     4     256       94.5    539.75      29.92
5     4     512       94.8    546.71      28.39
10    1     256       94.2    462.73      26.78
10    1     512       87.2    199.80      25.47
10    2     64        94.3    485.92      27.91
10    2     128       94.5    475.67      26.38
10    2     256       94.1    475.66      28.12
10    2     512       94.6    479.83      26.04
```

**Orin Key Insights**:
- **Sweet Spot**: 4MB workload with 128-256 threads per block achieves >94% improvement
- **Reliability**: Low variance in Green Context performance (23-35μs range)

#### Jetson Thor

**Performance Matrix** :
```
Load  Size  Threads   Avg%    Baseline    GC
Int   (MB)  /Block    Impr    Avg(μs)     Avg(μs)
----------------------------------------------------
5     1     64        73.5    44.41       11.75
5     1     128       69.9    39.59       11.93
5     1     256       61.8    31.78       12.14
5     1     512       73.5    38.54       10.19
5     4     64        95.3    211.91      9.97
5     4     128       95.2    271.31      12.95
5     4     256       94.1    226.10      13.32
5     4     512       96.2    278.20      10.62
5     16    64        99.1    1169.41     10.02
5     16    128       98.9    1192.72     12.83
5     16    256       99.3    1224.37     8.02
5     16    512       98.6    1096.32     15.40
20    4     64        99.3    1045.09     7.66
20    4     128       99.4    1092.23     6.67
20    4     256       98.7    1080.32     13.56
20    4     512       99.2    1056.79     8.94
20    16    64        99.7    4517.73     12.22
20    16    128       99.7    4500.81     13.12
20    16    256       99.7    4425.59     12.73
20    16    512       99.7    4477.69     13.71
80    8     64        99.8    7518.14     12.21
80    8     128       99.8    7321.67     12.52
80    8     256       99.8    7153.78     12.16
80    8     512       99.8    7273.29     13.58
```

**Jetson Thor Key Insights**:
- **Sweet Spot**: 16MB+ workload with load-intensity 20+ achieves >99% improvement
- **Excellent scaling**: Performance improves dramatically with workload size (1MB→16MB)
- **Consistent GC performance**: 8-15μs range regardless of baseline variability
- **High-end performance**: Up to 99.8% improvement with large workloads

#### RTX 6000 Ada Generation (142 SMs)

**Performance Matrix** :
```
Load  Size  Threads   Avg%    Baseline    GC
Int   (MB)  /Block    Impr    Avg(μs)     Avg(μs)
----------------------------------------------------
5     1     64        2.7     3.61        3.51
5     1     128       -0.8    3.35        3.38
5     1     256       10.8    4.08        3.64
5     1     512       6.3     3.48        3.26
5     2     64        55.8    9.15        4.05
5     2     128       46.0    8.18        4.41
5     2     256       58.4    10.21       4.25
5     2     512       55.5    9.70        4.31
5     4     64        81.4    24.22       4.51
5     4     128       80.8    24.15       4.63
5     4     256       80.6    22.84       4.44
5     4     512       81.6    23.40       4.31
5     8     64        93.8    70.23       4.35
5     8     128       93.3    68.74       4.59
5     8     256       93.0    64.91       4.53
5     8     512       92.0    57.69       4.61
5     16    64        96.6    154.00      5.26
5     16    128       96.3    143.26      5.36
5     16    256       96.6    146.40      4.93
5     16    512       95.9    138.77      5.74
10    8     64        95.7    115.27      4.96
10    8     128       95.5    108.94      4.95
10    8     256       95.8    110.68      4.67
10    8     512       95.6    112.37      4.98
10    16    64        97.9    286.67      5.93
10    16    128       97.5    271.31      6.80
10    16    256       97.7    278.56      6.49
10    16    512       97.8    280.72      6.06
20    8     64        97.8    266.16      5.74
20    8     128       97.9    266.65      5.59
20    8     256       97.8    261.38      5.89
20    8     512       97.7    268.53      6.08
40    1     64        97.8    201.41      4.34
40    4     64        97.1    210.54      6.08
40    4     128       97.1    201.47      5.85
40    4     256       97.3    215.81      5.89
40    4     512       96.9    189.12      5.87
```

**RTX 6000 Ada Key Insights**:
- **Sweet Spot**: 8-16MB workload with load-intensity 10+ achieves >95% improvement
- **Threshold Effect**: Minimal benefits below 2MB workload, dramatic improvements above 4MB
- **Reliability**: Low variance in Green Context performance (4-6μs range)

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

4. **CUDA linking errors**:
   ```
   /usr/bin/ld: cannot find /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcudart.so: No such file or directory
   /usr/bin/ld: cannot find /usr/local/cuda-12.8/targets/sbsa-linux/lib/libcupti.so: No such file or directory
   ```

   **Solution**: Clear holohub cache and refresh base image:
   ```bash
   ./holohub clear-cache
   docker rmi nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu
   ```
   Then retry the benchmark command. This resolves CUDA version path mismatches from stale cached layers.

### Performance Tuning

- **Low contention scenarios**: Increase `--load-intensity` to 20-100
- **More stable results**: Use `--samples 1000+` (default)
- **Large GPUs**: Increase `--workload-size` to 16-32MB
- **Memory-limited systems**: Reduce `--workload-size` to 4-8MB
- **Different thread configurations**: Adjust `--threads-per-block` (256, 512, 1024)
- **Specific testing**: Use `--mode baseline` or `--mode green-context` for focused testing

---
