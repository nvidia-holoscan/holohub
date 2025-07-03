# Holoscan Advanced Network library Changelog

## 🚧 holoscan-networking 0.2

**Release Date**: TBD

### Breaking

- Removed `get_port_from_ifname` and `address_to_port`. Use `get_port_id` instead.
- Removed the `adv_network_tx` (`AdvNetworkOpTx`) and `adv_network_rx` operators. See https://github.com/nvidia-holoscan/holohub/pull/743. Instead:
  - Initialize your NIC with `advanced_network::adv_net_init` in your application.
  - In your custom Tx operator, replace `op_output.emit` with `advanced_network::send_tx_burst`.
  - In your custom Rx operator, replace `op_input.receive` with `advanced_network::get_rx_burst`, also specifying the port and queue ids.

### Non-Breaking

Common:
- Added `get_port_id` to get the port ID for a given PCIe address or config name.
- Added `get_num_rx_queues` to get the number of RX queues for a given port.

-   **Enhanced Network Benchmarking:**
    -   The Advanced Networking Benchmark application now supports UDP port ranges (e.g., "5000-5010") for more flexible traffic distribution across queues.
    -   Added a new configuration (`adv_networking_bench_default_tx_rx_multi_q_hds.yaml`) for benchmarking multi-queue transmit/receive scenarios with Header-Data Split.
    -   Added option to disable the internal packet reordering kernel (`reorder_kernel: false`).
-   **Improved Advanced Network Core:**
    -   Fix crash when using a batch_size smaller than 128
    -   Support with per-queue timeouts, ensuring balanced processing even with uneven packet distribution or bursty traffic.
    -   Added configurable metadata buffer pools (`tx_meta_buffers`, `rx_meta_buffers`) for better resource tuning in high-rate scenarios.


## 🚀 holoscan-networking 0.1

**Release Date**: March 14, 2025

### Breaking

- Rename `doca` manager to `gpunetio`. See https://github.com/nvidia-holoscan/holohub/pull/707.
- Changed include headers and cmake target names. See https://github.com/nvidia-holoscan/holohub/pull/711.
- Interface port IDs are now identified at runtime and can't be specified in the YAML config. See https://github.com/nvidia-holoscan/holohub/pull/715.
- Refactored YAML config schema. See https://github.com/nvidia-holoscan/holohub/pull/720 and [docs](https://nvidia-holoscan.github.io/holohub/tutorials/high_performance_networking/#51-understand-the-configuration-parameters).
- Large namespace, types and API renaming. See https://github.com/nvidia-holoscan/holohub/pull/737.
- `adv_net_free_all_pkts_and_burst` is replaced by `adv_net_free_all_pkts_and_burst_rx` and `adv_net_free_all_pkts_and_burst_tx`. See https://github.com/nvidia-holoscan/holohub/pull/735.

### Non-Breaking

- Utility:
  - Added script to do basic system checks. See [docs](https://nvidia-holoscan.github.io/holohub/tutorials/high_performance_networking/#3-optimal-system-configurations).
- Common:
  - Switch to flow_isolation mode by default. See https://github.com/nvidia-holoscan/holohub/pull/624.
  - Fixed lockup with low buffer_size. See https://github.com/nvidia-holoscan/holohub/pull/735.
  - Support multiple queues per core. See https://github.com/nvidia-holoscan/holohub/pull/638.
  - Avoid crash on DPDK init failures. See https://github.com/nvidia-holoscan/holohub/pull/725.
  - Added stat reporting. See https://github.com/nvidia-holoscan/holohub/pull/735.
- GPUNetIO manager:
  - Disable rx persistent kernel by default. See https://github.com/nvidia-holoscan/holohub/pull/710.
- Benchmarking apps:
  - Remove unrelated configuration options.
  - Improve sample field names and values. See https://github.com/nvidia-holoscan/holohub/pull/727.
