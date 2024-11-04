<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# File Reader Operator

## Overview

The file reader operator is a simple IQ data source for downstream signal
processing applications.

## Description

During initialization, the operator loads raw IQ data from a user-provided
file into a MatX tensor. Then, it uses a user-provided sample rate to
send bursts of data on its output.

This operator will loop over the file data continuously.

### Data Format

Currently, the file reader supports ingesting complex, signed, 16-bit
integer data (i.e. 16-bit I, 16-bit Q). Once ingested, the file reader
will cast (with scaling) to 32-bit float values between -1.0 and +1.0
for more precise downstream computations.

## Requirements

- [MatX](https://github.com/NVIDIA/MatX)

## Example Usage

To make the file reader send at the configured sample rate, use Holoscan's
`PeriodicCondition` while creating the operator:

```cpp
auto file_reader_rate_hz = std::to_string(
    from_config("file_reader.sample_rate_sps").as<uint64_t>()
    / from_config("file_reader.burst_size").as<uint64_t>())
    + std::string("Hz");

auto fileReaderOp = make_operator<ops::FileReader>(
    "fileReader",
    from_config("file_reader"),
    make_condition<PeriodicCondition>("periodic-condition",
                                      Arg("recess_period") = file_reader_rate_hz));
```

For an full example, see the
[`psd_pipeline_sim`](../../applications/psd_pipeline_sim) application.

## Configuration

The file reader takes in a few parameters that determine how data is
sent and what metadata to pass along:

```yaml
file_reader:
  file_name: IQ_Data_8MSPS.dat
  burst_size: 1280
  sample_rate_sps: 8000000
  stream_id: 1
  rf_ref_freq_hz: 100000000
  bandwidth_hz: 6000000
  ref_level_dbm: 0
  gain_db: -10
```

- `file_name`: Name/path of the file from which to ingest data
- `burst_size`: Number of samples to send in one invocation of `compute()`
- `sample_rate_sps`: Rate at which the samples should be emitted
                     (also passed along in metadata)
- `stream_id`: Stream ID to pass along in the metadata
- `rf_ref_freq_hz`: Tune frequency to pass along in the metadata
- `bandwidth_hz`: Bandwidth to pass along in the metadata
- `ref_level_dbm`: Reference level to pass along in the metadata
- `gain_db`: Gain to pass along in the metadata
