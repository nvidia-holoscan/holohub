# GPU-resident Display Operator

GPU-resident display operator is built on top of the cuDisp DRM/CUDA display library.
Supports front-buffer rendering and G-SYNC (VRR) continuous flip mode. The operator can only work in exclusive display mode where no other display compositors are running. Turn off the default display manager, for example, on IGX systems, with `sudo service display-manager stop`.

## Building

```bash
./holohub build display_gpu_resident
```

### Building with debug prints

Enable verbose cuDisp debug output at build time with the `CUDISP_DEBUG` option:

```bash
./holohub build display_gpu_resident --configure-args='-DCUDISP_DEBUG=ON'
```

### Building with benchmarking

Enable present-thread benchmarking with the `CUDISP_BENCH` option:

```bash
./holohub build display_gpu_resident --configure-args='-DCUDISP_BENCH=ON'
```

### Building with tests

```bash
./holohub build imx274_gpu_resident --configure-args='-DBUILD_TESTING=ON'
```

All build options can be combined:

```bash
./holohub build imx274_gpu_resident --configure-args='-DBUILD_TESTING=ON -DCUDISP_DEBUG=ON -DCUDISP_BENCH=ON'
```

The same flags apply when building an application that depends on this operator:

```bash
./holohub build imx274_gpu_resident --configure-args='-DCUDISP_BENCH=ON'
```

## Running with benchmarking

Benchmark output is produced only in continuous flip mode (GPU-driven present thread).
This mode is active when front-buffer rendering is disabled. For `imx274_gpu_resident`,
pass `--gsync` to enable it:

```bash
CUDISP_BENCH_PRESENT=1 ./build/imx274_gpu_resident/applications/imx274_gpu_resident/cpp/imx274_gpu_resident --camera-mode 0 --gsync
```

!!! note
    `./holohub run imx274_gpu_resident` also works, but the application may not shut down gracefully.

## Environment variables

### Benchmarking

The following variables are available when the operator is built with `CUDISP_BENCH=ON`
and the present thread is active:

| Variable                      | Description                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `CUDISP_BENCH_PRESENT`        | Set to `1` to enable present-thread timing and counters. No benchmark output is produced when unset or `0`.                      |
| `CUDISP_BENCH_EVERY_SUCCESS`  | Set to `1` to print a line for every successful present with slot index and latency.                                             |
| `CUDISP_BENCH_REPORT_INTERVAL`| Number of successful presents between summary lines (default: `120`). Each summary reports success count, avg/min/max latency in microseconds, and ebusy/error counts. |

Example:

```bash
export CUDISP_BENCH_PRESENT=1
export CUDISP_BENCH_EVERY_SUCCESS=1
export CUDISP_BENCH_REPORT_INTERVAL=30
./build/.../imx274_gpu_resident --camera-mode 0 --gsync
```

### General

| Variable             | Description                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `CUDISP_DRM_DEVICE`  | Override the DRM device. Accepts a card number (e.g. `1` for `/dev/dri/card1`) or a full path.          |

## Tests

Test binaries are built when `-DBUILD_TESTING=ON` is passed and are placed under
`build/imx274_gpu_resident/operators/display_gpu_resident/cudisp/tests/`.

### Functional tests

Validate platform DRM/GBM/Vulkan capabilities. These do **not** link against or call the cuDisp library.

| Binary                 | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `drm_basic_test`       | DRM modeset, atomic commit, overlay, scaling, VRR, alpha, blend, rotation |
| `gbm_drm_test`         | GBM buffer allocation and DRM scanout with configurable flags            |
| `buffer_pipeline_test` | GBM to Vulkan to CUDA buffer pipeline with CPU or CUDA fill              |
| `format_test`          | DRM pixel format support (all DRM formats, GBM + dumb buffer fallback)   |

### API validation

GTest-based suite that validates cuDisp error handling and attribute parsing.

| Binary                | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `api_validation_test` | Attribute validation, error paths, unsupported feature detection      |

### Integration tests and demo

Exercise cuDisp host and GPU present paths with visual output.

| Binary              | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `host_present_test` | Host-driven present with CUDA-rendered patterns           |
| `perf_test`         | Latency and throughput benchmarks for host and GPU present |
| `cudisp_demo`       | Standalone GPU-present demo with animated color bars      |

### Running tests

```bash
# Functional DRM test
./build/imx274_gpu_resident/operators/display_gpu_resident/cudisp/tests/drm_basic_test

# API validation (GTest)
./build/imx274_gpu_resident/operators/display_gpu_resident/cudisp/tests/api_validation_test

# Performance test (GPU present, 10 seconds)
./build/imx274_gpu_resident/operators/display_gpu_resident/cudisp/tests/perf_test --present-mode gpu --duration 10

# Demo (GPU present, animated bars)
./build/imx274_gpu_resident/operators/display_gpu_resident/cudisp/tests/cudisp_demo --duration 10
```

All test binaries accept `--help` for a full option listing.

See also the [cuDisp README](cudisp/README.md) for library-level details.
