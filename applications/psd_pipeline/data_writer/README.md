<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# Data Writer Operator

## Overview

Writes binary data from its input to an output file. This operator is intened
for use as a debugging aid.

## Description

The data writer operator takes in a `std::tuple<tensor_t<complex, 2>, cuda_stream_t>`,
copies the data to a host tensor, then writes the data out to a binary file.

The file path is determined based on input metadata with the following
keys:

1. `channel_number` (default `0`)
2. `bandwidth_hz` (default `0.0`)
3. `rf_ref_freq_hz` (default `0.0`)

With this, it creates: `data_writer_out_ch{channel_number}_bw{bandwidth_hz}_freq{rf_ref_freq_hz}.dat`.

## Requirements

- [MatX](https://github.com/NVIDIA/MatX) (dependency - assumed to be installed on system)

## Configuration

The data writer operator takes in a few parameters:

```yaml
data_writer:
  burst_size: 1280
  num_bursts: 625
```

- `burst_size`: Number of samples contained in each burst
- `num_bursts`: Number of bursts to process at once

## Example Usage

For an example of how to use this operator, see the
[`psd_pipeline`](../../psd_pipeline) application.

Usually, you'd just want to write one burst of data to a file. To
do that, you could use a `CountCondition` to limit the number of
times this operator runs:

```cpp
auto dataWriterOp = make_operator<ops::DataWriter>(
    "dataWriterOp",
    make_condition<CountCondition>(1));
```
