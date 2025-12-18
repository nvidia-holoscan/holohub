<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# VITA 49 Power Spectral Density (PSD)

## Overview

The VITA 49 Power Spectral Density (PSD) application takes in a VITA49 data stream from the advanced network
operator, then performs an FFT, PSD, and averaging operation before
generating a VITA 49.2 spectral data packet which gets sent to a
destination UDP socket.

![Diagram of the PSD pipeline showing each operator and the flow
  of GPU and CPU data](./imgs/psd_pipeline_diagram.png)

### Acronyms

| Acronym | Meaning                                                            |
| ------- | ------------------------------------------------------------------ |
| FFT     | Fast Fourier Transform                                             |
| NIC     | Network Interface Card                                             |
| PSD     | Power Spectral Display                                             |
| VITA 49 | Standard for interoperability between RF (radio frequency) devices |
| VRT     | VITA Radio Transport (transport-layer protocol)                    |

## Requirements

- ConnectX 6 or 7 NIC for GPUDirect RDMA with packet size steering
- [MatX](https://github.com/NVIDIA/MatX) (dependency - assumed to be installed on system)
- [vita49-rs](https://github.com/voyager-tech-inc/vita49-rs) (dependency)

## Configuration

> [!IMPORTANT]
> The settings in `config.yaml` need to be tailored to your system/radio.

Each operator in the pipeline has its own configuration section. The specific options
and their meaning are defined in each operator's own README:

1. [`advanced_network`](../../operators/advanced_network/README.md)
2. [`vita_connector`](./advanced_network_connectors/README.md)
3. [`fft`](../../operators/fft/README.md)
4. [`high_rate_psd`](../../operators/high_rate_psd/README.md)
5. [`low_rate_psd`](../../operators/low_rate_psd/README.md)
6. [`vita49_psd_packetizer`](../../operators/vita49_psd_packetizer/README.md)

There is also one option specific to this application:

1. `num_psds`: Number of PSDs to produce out of the pipeline before exiting.
               Passing `-1` here will cause the pipeline to run indefinitely.

### Metadata

This pipeline leverages Holoscan's operator metadata dictionaries to pass
VITA 49-adjacent metadata through the pipeline.

Each operator in the pipeline adds its own metadata to the dictionary.
At the end of the pipeline, the packetizer operator uses the metadata
to construct VITA 49 context packets to send alongside the spectral data.

### Memory Layout

The ANO operates using memory regions that it directs data to. Since VITA49
is somewhat unusual in that signal data packets and context packets arrive
at the same IP/port, we want to use the ANO's packet length steering feature
to drop packets in the appropriate memory region.

First, we want to define our memory regions:

1. A region for any packets that don't match any of our flows [CPU]
2. A region for _frame_ headers (i.e. Ethernet + IP + UDP) [CPU]
   - These headers are not currently used, so this memory region is
     essentially acting as a `/dev/null` sink.
3. A region for each channel's _VRT_ headers [CPU]
   - We need these headers to grab things like stream ID and
     timestamp, but don't need that information in the GPU processing,
     so make this a CPU region.
4. A region for each channel's VRT signal data [GPU]
   - These are the raw IQ samples from our radio - we want these to
     land in GPU memory via GPUDirect RDMA.
5. A region for _all_ channels' VRT context data [CPU]
   - We need the whole packet in the CPU to fill out our metadata
     map for downstream processing/packet assembly.

When an individual packet comes in, the ANO will try to match it
against the defined flows. So, for our data packets, we want to
define a queue like this:

```yaml
            flows:
              - name: "Data packets"
                id: 0
                action:
                  type: queue
                  id: 1
                match:
                  # Match with the port your SDR is sending to and the
                  # length of the signal data packets
                  udp_src: 4991
                  udp_dst: 4991
                  ipv4_len: 4148
```

This is saying "if a UDP packet with IPv4 length 4,148 comes in on port
4991, send it to the queue with ID 1". Now, if we look at our queue with
ID 1, we see:

```yaml
              - name: "Data"
                id: 1
                cpu_core: 5
                batch_size: 12500
                memory_regions:
                  - "Headers_RX_CPU"
                  - "VRT_Headers_RX_CPU"
                  - "Data_RX_GPU"
```

When multiple `memory_regions` are specified like this, it means
that each packet should be split based on the memory region size. In this
case, `Headers_RX_CPU` has `buf_size: 42` (the size of the frame header),
`VRT_Headers_RX_CPU` has `buf_size: 20` (the size of the VRT header),
and `Data_RX_GPU` has `buf_size: 4100` (the remaining size of the data
packet). These numbers may be different depending on the packet size of
your radio!

`batch_size: 12500` tells the ANO to batch up 12,500 packets before sending
the data to downstream operators. In our case, 12,500 packets represents
100ms worth of data and that's how much we want to process on each run of
the pipeline.

### Multiple Channels

When working with multiple channels, this pipeline expects all context
packets (from every channel) to flow to one queue, but each data channel
flows to its own queue.

The connector operator also makes the following assumptions:

1. All context packets flow to queue `id: 0`.
2. All context packets flow ID matches its channel (e.g., flow ID `1`
   is for context packets from channel `1`).
3. All data packets arrive on a queue ID one greater than its channel
   (e.g., queue ID `1` is for channel `0` data).
4. The `batch_size` of the context queue is equal to the number of
   channels.


### Ingest NIC

The PCIe address of your ingest NIC needs to be specified in `config.yaml`.

```yaml
    interfaces:
      - name: sdr_data
        address: 0000:17:00.0
```

You can find the addresses of your devices with: `lshw -c network -businfo`:

```
# lshw -c network -businfo
Bus info          Device     Class          Description
=======================================================
pci@0000:03:00.0  eth0       network        I210 Gigabit Network Connection
pci@0000:06:00.0  eno1       network        Aquantia Corp.
pci@0000:51:00.0  ens3f0np0  network        MT2910 Family [ConnectX-7]
pci@0000:51:00.1  ens3f1np1  network        MT2910 Family [ConnectX-7]
usb@1:14.2        usb0       network        Ethernet interface
```

In this example, if you wanted to use the `ens3f1np1` interface, you'd pass
`0000:51:00.1`.

## Build & Run
1. **Build** the development container in two steps:
   ```bash
   # Build the ANO dev container
   ./holohub build-container advanced_network --docker-file ./operators/advanced_network/Dockerfile

   # Add the psd-pipeline deps
   ./holohub build-container psd_pipeline --base-img holohub:ngc-v3.1.0-dgpu --img holohub-psd-pipeline:ngc-v3.1.0-dgpu
   ```
2. **Launch** the development container with the command:
   ```bash
   ./holohub run-container psd_pipeline --no-docker-build --docker-opts="-u root --privileged" --img holohub-psd-pipeline:ngc-v3.1.0-dgpu
   ```

Once you are in the dev container:
1. **Build** the application using:
    ```bash
    ./holohub build psd_pipeline
    ```
2. **Run** the application using:
    ```bash
    ./holohub run psd_pipeline --local --no-local-build --run-args="config.yaml"
    ```
