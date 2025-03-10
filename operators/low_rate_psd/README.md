<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# Low Rate PSD Operator

## Overview

PSD accumulator/averager.

## Description

The low rate PSD operator...
- takes in `num_averages` tensors of float data,
- takes an average of all the accumulated tensors,
- performs a 10 * log10() operation on the average,
- clamps data to 8-bit integer boundaries,
- casts to signed 8-bit integers,
- emits the resultant tensor

## Requirements

- [MatX](https://github.com/NVIDIA/MatX) (dependency)

## Example Usage

For an example of how to use this operator, see the
[`psd_pipeline`](../../applications/psd_pipeline) application.

## Multiple Channels

The zero-indexed `channel_number` key will be looked up in [`metadata()`](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#dynamic-application-metadata)
on each `compute()` run. If no value is found, the default channel number is `0`.

## Configuration

The low rate PSD operator takes two parameters:

```yaml
low_rate_psd:
  burst_size: 1280
  num_averages: 625
  num_channels: 1
```

- `burst_size`: Number of samples to process on each invocation of `compute()`
- `num_channels`: Number of channels for which to allocate memory
- `num_averages`: How many PSDs to accumulate before averaging and emitting.
