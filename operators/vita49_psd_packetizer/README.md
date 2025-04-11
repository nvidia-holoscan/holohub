<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# VITA49 PSD Packetizer

## Overview

Generate VITA 49.2 spectral data packets from incoming data.

## Description

This operator will take in PSD data computed by upstream operators
and format it into VITA 49.2 Spectral Data packets.

After creating the VRT packets, it will send the packets to the configured
UDP IP/port.

## Requirements

- [MatX](https://github.com/NVIDIA/MatX) (dependency - assumed to be installed on system)
- [Rust](https://www.rust-lang.org/) (language dependency)
- [vita49](https://github.com/voyager-tech-inc/vita49-rs) (Rust library dependency)

Note: this operator depends on a Rust component. The `Dockerfile` provided
in this directory will install Rust in the dev container from the official
Ubuntu repos.

## Multiple Channels

If multiple channels are configured, the packetizer will use the base port
in the configuration and add the channel index. So, with `base_dest_port: 4991`,
channel `0` would send data to `4991`, but channel `1` would send data to `4992`.

The zero-indexed `channel_number` key will be looked up in [`metadata()`](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#dynamic-application-metadata)
on each `compute()` run. If no value is found, the default channel number is `0`.

## Example Usage

For an example of how to use this operator, see the
[`psd_pipeline`](../../applications/psd_pipeline) application.

## Configuration

The packetizer takes the following parameters:

```yaml
vita49_psd_packetizer:
  burst_size: 1280
  num_channels: 1
  dest_host: 127.0.0.1
  base_dest_port: 4991
  manufacturer_oui: 0xFF5646
  device_code: 0x80
```

- `burst_size`: Number of samples to process in each burst
- `num_channels`: Number of channels for which to allocate memory
- `dest_host`: Destination host
- `base_dest_port`: Base destination UDP port
- `manufacturer_oui`: Manufacturer identifier to embed in the context packets
- `device_code`: Device code to embed in the context packets
