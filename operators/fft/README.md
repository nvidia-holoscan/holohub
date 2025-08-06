<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# Fast Fourier Transform (FFT) Operator

## Overview

A thin wrapper over the MatX [`fft()`](https://nvidia.github.io/MatX/api/dft/fft/fft.html)
executor.

## Description

The FFT operator takes in a tensor of complex float data, performs an FFT,
and emits the resultant tensor.

## Requirements

- [MatX](https://github.com/NVIDIA/MatX) (dependency - assumed to be installed on system)

## Example Usage

For an example of how to use this operator, see the
[`psd_pipeline`](../../applications/psd_pipeline) application.

## Multiple Channels

The zero-indexed `channel_number` key will be looked up in [`metadata()`](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#dynamic-application-metadata)
on each `compute()` run. If no value is found, the default channel number is `0`.

## Configuration

The FFT operator takes in a few parameters:

```yaml
fft:
  burst_size: 1280
  num_bursts: 625
  num_channels: 1
  spectrum_type: 1
  averaging_type: 1
  window_time: 0
  window_type: 0
  transform_points: 1280
  window_points: 1280
  resolution: 6250
  span: 8000000
  weighting_factor: 0
  f1_index: -640
  f2_index: 639
  window_time_delta: 0
```

The only parameters that actually impacts FFT computation at this point
are the `burst_size` and `num_bursts` params. The rest of the parameters
are simply passed along in the metadata.

- `burst_size`: Number of samples to process in each burst
- `num_bursts`: Number of bursts to process at once
- `num_channels`: Number of channels for which to allocate memory
- `spectrum_type`: VITA 49.2 spectrum type to pass along in metadata
- `spectrum_type`: VITA 49.2 spectrum type to pass along in metadata
- `averaging_type`: VITA 49.2 averaging type to pass along in metadata
- `window_time`: VITA 49.2 window time to pass along in metadata
- `window_type`: VITA 49.2 window type to pass along in metadata
- `transform_points`: Number of FFT points to take and VITA 49.2 transform points to pass along in metadata
- `window_points`: VITA 49.2 window points to pass along in metadata
- `resolution`: VITA 49.2 resolution to pass along in metadata
- `span`: VITA 49.2 span to pass along in metadata
- `weighting_factor`: VITA 49.2 weighting factor to pass along in metadata
- `f1_index`: VITA 49.2 F1 index to pass along in metadata
- `f2_index`: VITA 49.2 F2 index to pass along in metadata
- `window_time_delta`: VITA 49.2 window time delta to pass along in metadata
