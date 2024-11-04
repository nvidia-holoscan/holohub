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

- [MatX](https://github.com/NVIDIA/MatX)
- [vrtgen](https://github.com/Geontech/vrtgen)

## Example Usage

For an example of how to use this operator, see the
[`psd_pipeline_sim`](../../applications/psd_pipeline_sim) application.

## Configuration

The packetizer takes the following parameters:

```yaml
vita49_psd_packetizer:
  dest_host: 127.0.0.1
  dest_port: 4991
  manufacturer_oui: 0xFF5646
  device_code: 0x80
```

- `dest_host`: Destination host
- `dest_port`: Destination UDP port
- `manufacturer_oui`: Manufacturer identifier to embed in the context packets
- `device_code`: Device code to embed in the context packets
