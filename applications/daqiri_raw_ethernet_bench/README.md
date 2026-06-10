# DAQIRI Raw Ethernet Benchmark

> [!TIP]
> Review the [High Performance Networking tutorial](/tutorials/high_performance_networking/README.md) for guided
> instructions to configure your system and test DAQIRI.

This sample application measures raw DAQIRI DPDK transmit and receive throughput using the single
`daqiri_raw_ethernet_bench_default_tx_rx.yaml` configuration. The benchmark sends UDP packets from one
configured interface and receives them on another configured interface.

The performance of this application depends heavily on a properly-configured system and tuning parameters
that are acceptable for the workload. To configure the system, see the
[High Performance Networking documentation](/tutorials/high_performance_networking/README.md).

## Configuration

Edit `daqiri_raw_ethernet_bench_default_tx_rx.yaml` and replace the placeholder interface PCIe addresses,
destination MAC address, and IP addresses with values for your system.

The receiver uses `bench_rx` settings:

- `batch_size`: packet count for one processing batch.
- `max_packet_size`: maximum received packet size, including headers.
- `header_size`: bytes to skip before the payload.

The transmitter uses `bench_tx` settings:

- `batch_size`: packet count for one transmit burst.
- `payload_size`: payload bytes after all L2-L4 headers.
- `eth_dst_addr`, `ip_src_addr`, `ip_dst_addr`, `udp_src_port`, and `udp_dst_port`: packet header fields.

## Requirements

This application requires DAQIRI and a system configured for DAQIRI DPDK raw packet IO.
The HoloHub networking container builds DAQIRI from
[`NVIDIA/daqiri`](https://github.com/NVIDIA/daqiri), installs it as the `daqiri`
system package in the image, and exposes it from `/opt/daqiri`. To use a specific
DAQIRI revision, pass a Docker build argument such as
`--build-args "--build-arg DAQIRI_REF=<tag-or-sha>"`.

## Build Instructions

```bash
./holohub build daqiri_raw_ethernet_bench --language=cpp
```

## Run Instructions

```bash
./holohub run daqiri_raw_ethernet_bench --language=cpp \
  --docker-opts="-u root --privileged" \
  --run-args="daqiri_raw_ethernet_bench_default_tx_rx.yaml"
```

To run manually inside the container:

```bash
./holohub run-container daqiri_raw_ethernet_bench \
  --language cpp \
  --docker-opts="-u root --privileged -w /workspace/holohub/"

./holohub build daqiri_raw_ethernet_bench --language=cpp
./build/daqiri_raw_ethernet_bench/applications/daqiri_raw_ethernet_bench/cpp/daqiri_raw_ethernet_bench \
  daqiri_raw_ethernet_bench_default_tx_rx.yaml
```

## Test Instructions

Build with testing enabled:

```bash
./holohub build daqiri_raw_ethernet_bench --configure-args="-D BUILD_TESTING:BOOL=ON"
```

Then run the benchmark tests from the build directory:

```bash
./holohub run-container daqiri_raw_ethernet_bench \
  --docker-opts "-u 0 --privileged -w /workspace/holohub/build/daqiri_raw_ethernet_bench/" \
  -- ctest --output-on-failure
```
