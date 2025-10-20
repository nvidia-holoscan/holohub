# Low Rate PSD Operator

A Power Spectral Density (PSD) accumulator and averager operator.

## Overview

The **Low Rate PSD Operator** is a utility for computing and averaging the Power Spectral Density (PSD) of input signals. PSD is a fundamental tool in signal processing for analyzing the power distribution of a signal across frequency components. This operator is designed to efficiently accumulate and average PSDs over multiple bursts of input data, making it suitable for applications such as spectrum monitoring, signal diagnostics, and real-time analysis in embedded or high-throughput environments.

The Low Rate PSD Operator performs the following steps:

1. **Input Accumulation:** Receives `num_averages` tensors containing float data representing signal samples or pre-computed PSDs.
2. **Averaging:** Computes the average of all accumulated tensors to smooth out noise and fluctuations.
3. **Logarithmic Scaling:** Applies a `10 * log10()` operation to convert the averaged power values to decibel (dB) scale, which is standard for PSD representation.
4. **Clamping:** Restricts the data to the range of 8-bit signed integers to ensure compatibility and efficient storage.
5. **Casting:** Converts the clamped values to signed 8-bit integers.
6. **Emission:** Outputs the final tensor for downstream processing or analysis.

## Requirements

- [MatX](https://github.com/NVIDIA/MatX): Required for tensor operations (assumed to be installed on your system).

## Example Usage

For a practical example, see the [`psd_pipeline`](../../applications/psd_pipeline) application.

### Basic Workflow

1. Configure the operator parameters (see below).
2. Feed input tensors (float32 arrays) to the operator.
3. Collect the output signed 8-bit integer tensor representing the averaged PSD in dB.

## Multiple Channels

The operator supports processing multiple channels in parallel. The zero-indexed `channel_number` key is retrieved from [`metadata()`](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#dynamic-application-metadata) on each `compute()` invocation. If `channel_number` is not provided, the default is `0` (single channel).

## Configuration

Configure the operator in your application (e.g., YAML config):

```yaml
low_rate_psd:
  burst_size: 1280         # Number of samples processed per compute() call
  num_averages: 625        # Number of PSDs to accumulate before averaging
  num_channels: 1          # Number of signal channels to process
```

## Input/Output

- **Input:** Tensors of type `float32`, shape determined by `burst_size` and `num_channels`.
- **Output:** Tensor of type `int8`, representing the averaged PSD in dB scale, clamped to [-128, 127].

## Notes

- Ensure the input data is properly normalized and formatted as expected by the operator.
- The operator is optimized for performance and memory efficiency, leveraging MatX for tensor operations.
- For advanced use cases, refer to the Holoscan SDK documentation and the example pipeline linked above.
