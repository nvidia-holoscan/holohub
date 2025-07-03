# Advanced Networking Benchmark

> [!TIP]
> Review the [High Performance Networking tutorial](/tutorials/high_performance_networking/README.md) for guided
> instructions to configure your system and test the Advanced Network library.

This is a sample application to measure a lower bound on performance for the [Advanced Network
library](/operators/advanced_network/README.md) by receiving packets, optionally doing work on them, and freeing
the buffers. While only freeing the packets is an unrealistic workload, it's useful to see at a high level whether
the application is able to keep up with the bare minimum amount of work to do. The application contains both a
transmitter and receiver that are designed to run on different systems, and may be configured independently.

The performance of this application depends heavily on a properly-configured system and choosing the best
tuning parameters that are acceptable for the workload. To configure the system please see the documentation
[here](/tutorials/high_performance_networking/README.md). With the system tuned, the application performance
will be dictated by batching size and whether GPUDirect is enabled.

At this time both the transmitter and receiver are written to handle an Ethernet+IP+UDP packet with a
configurable payload. Other modes may be added in the future. Also, for simplicity, the transmitter and
receiver are configured to a single packet size.

## Transmit

The transmitter sends a UDP packet with an incrementing sequence of bytes after the UDP header. The batch
size configured dictates how many packets the benchmark operator sends to the NIC in each tick. Typically with
the same number of CPU cores the transmitter will run faster than the receiver, so this parameter may be used
to throttle the sender somewhat by making the batches very small.

## Receiver

The receiver receives the UDP packets in either CPU-only mode, header-data split mode, or GPU-only mode.
- CPU-only mode will receive the packets in CPU memory, copy the payload contents to a host-pinned staging buffer,
  and free the buffers.
- Header-data split mode: the user may configure separate memory regions for the header and data. The header is
  sent to the CPU, and all bytes afterwards are sent to the GPU. Header-data split should achieve higher
  rates than CPU mode since the amount of data to the CPU can be orders of magnitude lower compared to running
  in CPU-only mode.
- GPU-only mode: all bytes of the packets are received in GPU memory.

### Configuration

The application can be configured to do either Rx, Tx, or both, using different configuration files,
found in this directory.

#### Receive Configuration

- `header_data_split`: bool
  Turn on GPUDirect header-data split mode
- `batch_size`: integer
  Size in packets for a single batch. This should be a multiple of the advanced_network queue batch size.
  A larger batch size consumes more memory since any work will not start unless this batch size is filled. Consider
  reducing this value if errors are occurring.
- `max_packet_size`: integer
  Maximum packet size expected. This value includes all headers up to and including UDP.

#### Transmit Configuration

- `batch_size`: integer
  Size in packets for a single batch. This batch size is used to send to the NIC, and
  will loop sending that many packets for each burst.
- `payload_size`: integer
  Size of the payload to send after all L2-L4 headers

### Requirements

This application requires all configuration and requirements from the [Advanced Network library](/operators/advanced_network/README.md).

### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

```bash
./holohub run adv_networking_bench
```

### Run Instructions

First, go in your `build` or `install` directory, then:

```bash
./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_default_tx_rx.yaml
```

With DOCA:

```bash
./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_gpunetio_tx_rx.yaml
```

With RIVERMAX RX:

```bash
./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_rmax_rx.yaml
```
