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
- casts to signed 8-bit integers,
- emits the resultant tensor

## Requirements

- [MatX](https://github.com/NVIDIA/MatX)

## Example Usage

For an example of how to use this operator, see the
[`psd_pipeline_sim`](../../applications/psd_pipeline_sim) application.

## Configuration

The low rate PSD operator takes two parameters:

```yaml
low_rate_psd:
  burst_size: 1280
  num_averages: 625
```

- `burst_size`: Number of samples to process on each invocation of `compute()`
- `num_averages`: How many PSDs to accumulate before averaging and emitting.
