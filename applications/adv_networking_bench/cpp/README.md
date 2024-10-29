# Advanced Networking Benchmark

This is a simple application to measure a lower bound on performance for the advanced networking operator
by receiving packets, optionally doing work on them, and freeing the buffers. While only freeing the packets is
an unrealistic workload, it's useful to see at a high level whether the application is able to keep up with
the bare minimum amount of work to do. The application contains both a transmitter and receiver that are
designed to run on different systems, and may be configured independently.

The performance of this application depends heavily on a properly-configured system and choosing the best
tuning parameters that are acceptable for the workload. To configure the system please see the documentation
for the advanced network operator. With the system tuned, the application performance will be dictated
by batching size and whether GPUDirect is enabled. 

At this time both the transmitter and receiver are written to handle an Ethernet+IP+UDP packet with a
configurable payload. Other modes may be added in the future. Also, for simplicity, the transmitter and
receiver must be configured to a single packet size.

## Transmit

The transmitter sends a UDP packet with an incrementing sequence of bytes after the UDP header. The batch
size configured dictates how many packets the benchmark operator sends to the advanced network operator
in each tick. Typically with the same number of CPU cores the transmitter will run faster than the receiver, 
so this parameter may be used to throttle the sender somewhat by making the batches very small.

## Receiver

The receiver receives the UDP packets in either CPU-only mode or header-data split mode. CPU-only mode
will receive the packets in CPU memory, copy the payload contents to a host-pinned staging buffer, and
freed. In header-data split mode the user may configure a split point where the bytes before that point
are sent to the CPU, and all bytes afterwards are sent to the GPU. Header-data split should achieve higher
rates than CPU mode since the amount of data to the CPU can be orders of magnitude lower compared to running
in CPU-only mode. 

### Configuration

The application is configured using a separate transmit and receive file. The transmit file is called
`adv_networking_bench_tx.yaml` while the receive is named `adv_networking_bench_rx.yaml`. Configure the
advanced networking operator on both transmit and receive per the instructions for that operator.

#### Receive Configuration

- `header_data_split`: bool
  Turn on GPUDirect header-data split mode
- `batch_size`: integer
  Size in packets for a single batch. This should be a multiple of the advanced network RX operator batch size.
  A larger batch size consumes more memory since any work will not start unless this batch size is filled. Consider
  reducing this value if errors are occurring.
- `max_packet_size`: integer
  Maximum packet size expected. This value includes all headers up to and including UDP.

#### Transmit Configuration

- `batch_size`: integer
  Size in packets for a single batch. This batch size is used to send to the advanced network TX operator, and 
  will loop sending that many packets for each burst.
- `payload_size`: integer
  Size of the payload to send after all L2-L4 headers 

### Requirements

This application requires all configuration and requirements from the advanced network operator.

### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

```bash
./dev_container build_and_run adv_networking_bench
```

### Run Instructions

First, go in your `build` or `install` directory, then for the transmitter run:

```bash
./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_default_tx.yaml
```

Or for the receiver:

```bash
./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_default_rx.yaml
```

With DOCA:

```bash
./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_doca_tx_rx.yaml
```

With RIVERMAX RX:

```bash
./build/applications/adv_networking_bench/cpp/adv_networking_bench adv_networking_bench_rmax_rx.yaml
```

<mark>For Holoscan internal reasons (not related to the DOCA library), build the Advanced Network Operator with `RX_PERSISTENT_ENABLED` set to 1 MAY cause problems to this application on the receive (process) side (receive hangs). If you experience any issue on the receive side, please read carefully in the Advanced Network Operator README how to solve this problem.</mark>
