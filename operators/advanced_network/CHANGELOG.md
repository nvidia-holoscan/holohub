# Holoscan Advanced Network Operator Changelog

## 🚧 holoscan-networking 0.2

**Release Date**: TBD

### Breaking

-

### Non-Breaking

-

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
- DPDK manager:
  - Support multiple queues per core. See https://github.com/nvidia-holoscan/holohub/pull/638.
  - Avoid crash on DPDK init failures. See https://github.com/nvidia-holoscan/holohub/pull/725.
  - Added stat reporting. See https://github.com/nvidia-holoscan/holohub/pull/735.
- GPUNetIO manager:
  - Disable rx persistent kernel by default. See https://github.com/nvidia-holoscan/holohub/pull/710.
- Benchmarking apps:
  - Remove unrelated configuration options.
  - Improve sample field names and values. See https://github.com/nvidia-holoscan/holohub/pull/727.
