# Simple Radar Pipeline

This demonstration walks the developer through building a simple radar signal processing pipeline, targeted towards detecting objects, with Holoscan. In this example, we generate random radar and waveform data, passing both through:

1. Pulse Compression
2. Moving Target Indication (MTI) Filtering
3. Range-Doppler Map
4. Constant False Alarm Rate (CFAR) Analysis

While this example generates 'offline' complex-valued data, it could be extended to accept streaming data from a phased array system or simulation via modification of the `SignalGeneratorOperator`.

The output of this demonstration is a measure of the number of pulses per second processed on GPU.

The main objectives of this demonstration are to:

- Highlight developer productivity in building an end-to-end streaming application with Holoscan and existing GPU-Accelerated Python libraries
- Demonstrate how to construct and connect isolated units of work via Holoscan operators, particularly with handling multiple inputs and outputs into an Operator
- Emphasize that operators created for this application can be reused in other ones doing similar tasks

## Running the Application

```bash
./holohub run simple_radar_pipeline
```

## Performance: Explicit cuFFT Plan Caching

When profiling with NVIDIA Nsight Systems, a large `cuModuleLoadData` call was observed at the start of every pipeline iteration. This indicated that cuFFT plans were being evicted from CuPy's internal LRU plan cache between ticks of `compute()`, causing the GPU driver to reload and recompile CUDA modules on each call.

The fix is to explicitly create and hold `cupy.cuda.cufft.Plan1d` objects as instance attributes on the operators that perform FFTs (`PulseCompressionOp` and `RangeDopplerOp`). Plans are created once on the first `compute()` call and reused on every subsequent tick via CuPy's `with plan:` context manager, which bypasses the LRU cache lookup entirely.
