<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# VITA 49 Connector Operator

## Overview

An operator to load ANO VRT packet data into MatX tensors for downstream
processing.

## Description

The VITA 49 connector takes in Advanced Network Operator bursts and does
a few things:

1. Accumulates a configurable number of packets
2. Parses VITA 49 context packets and assigns metadata map values for
   downstream processing
3. Byteswaps from network byte-order to little-endian
4. Casts incoming data from 16-bit complex integer to 32-bit complex float
   (scaling to -1.0 thru +1.0)

## Requirements

- [ANO](https://github.com/nvidia-holoscan/holohub/tree/main/operators/advanced_network)
  (and associated hardware)
- [MatX](https://github.com/NVIDIA/MatX) (dependency - assumed to be installed on system)

## Configuration

```yaml
vita_connector:
  num_complex_samples_per_packet: 1024
  num_packets_per_fft: 20
  num_ffts_per_batch: 625
  num_simul_batches: 2
  num_channels: 4
```

- `num_complex_samples_per_packet`: Number of complex samples contained in every VRT data packet
- `num_packets_per_fft`: Number of packets you'd like to process in each FFT
- `num_ffts_per_batch`: Number of FFTs you'd like to perform in one downstream run
- `num_simul_batches`: Number of simultaneous batches to process (ping-pong style)
- `num_channels`: Number of channels to support

These parameters impact the shape of the data tensor that is assembled for downstream
processing. In the example above, the VITA 49 connector would emit a 625x20480 sample
`tensor_t`.
