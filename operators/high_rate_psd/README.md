<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# High Rate PSD Operator

## Overview

A thin wrapper over the MatX [`abs2()`](https://nvidia.github.io/MatX/api/math/misc/abs2.html)
executor.

## Description

The high rate PSD operator...
- takes in a tensor of complex float data,
- performs a squared absolute value operation on the tensor: real(t)^2 + imag(t)^2,
- divides by the number of input elements
- emits the resultant tensor

## Requirements

- [MatX](https://github.com/NVIDIA/MatX) (dependency - assumed to be installed on system)

## Example Usage

For an example of how to use this operator, see the
[`psd_pipeline`](../../applications/psd_pipeline) application.

## Multiple Channels

The zero-indexed `channel_number` key will be looked up in [`metadata()`](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#dynamic-application-metadata)
on each `compute()` run. If no value is found, the default channel number is `0`.

## Configuration

The operator only takes the following parameters:

```yaml
high_rate_psd:
  burst_size: 1280
  num_bursts: 625
  num_channels: 1
```

- `burst_size`: Number of samples to process in each burst
- `num_bursts`: Number of bursts to process at once
- `num_channels`: Number of channels for which to allocate memory
