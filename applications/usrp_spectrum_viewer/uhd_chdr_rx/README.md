
# UHD CHDR Receiver Operator

## Overview

An operator to receive data packets streamed from USRP's (CHDR packets) and converts them into batched complex-sample tensors for downstream processing.

## Description

The UHD CHDR Receiver takes in DAQIRI packet bursts and does a few things:

1. Accumulates a configurable number of packets
2. Parses CHDR protocol headers and extracts the contained RF data
3. Casts incoming data from 16-bit complex integer to 32-bit complex float
   (scaling to -1.0 thru +1.0)

## Requirements

- [DAQIRI](https://github.com/NVIDIA/daqiri) (and associated hardware)
- [MatX](https://github.com/NVIDIA/MatX) (dependency - assumed to be installed on system)

## Configuration

```yaml
uhd_chdr_rx:
  interface_name: sdr_data
  num_complex_samples_per_packet: 1024
  num_packets_per_output: 20
  num_outputs_per_batch: 125
  num_buffered_batches: 2
  num_channels: 2
  log_packets: false
  log_data: false
```

- `interface_name`: Name of the RX port from the daqiri config
- `num_complex_samples_per_packet`: Number of complex samples contained in every CHDR data packet
- `num_packets_per_output`: Number of packets grouped into one emitted output row.
- `num_outputs_per_batch`: Number of output rows emitted together in one batch.
- `num_buffered_batches`: Maximum number of converted batches waiting per channel. When the queue reaches this limit, receive polling for that channel pauses until a completed batch is emitted.
- `num_channels`: Number of channels to support
- `log_packets`: Log the first packet of a burst to console
- `log_data`: Log the complex floating point data for the first packet of a burst to console

These parameters impact the shape of the data tensor that is assembled for downstream
processing. In the example above, the UHD CHDR receiver would emit a 125x20480 sample
`tensor_t`.
