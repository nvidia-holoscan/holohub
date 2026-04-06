# HSB RoCE Receiver Operator with DOCA GPUNetIO

This [GPU-resident Holoscan operator](https://docs.nvidia.com/holoscan/sdk-user-guide/gpu_resident.html) receives messages over RoCE (RDMA over Converged Ethernet)
directly to GPU memory using NVIDIA DOCA GPUNetIO for GPU-side completion queue
(CQ) polling. By performing CQ polling entirely on the GPU, this operator
eliminates the CPU from the RDMA receive critical path, enabling fully
GPU-resident frame reception.

This operator is an alternative to the no-host-metadata (NMD)-based
[`hsb_roce_receiver_nmd`](../hsb_roce_receiver_nmd/) receiver. It can be used
as a drop-in replacement when the `--docagpunetio` flag is passed to the
`imx274_gpu_resident` application.

## Overview

The `DocaRoceReceiverOp` is a GPU-resident Holoscan operator that:

- Receives messages via RDMA directly to GPU memory
- Polls the RDMA Completion Queue (CQ) from a CUDA kernel running inside a
  Holoscan SDK GPU-resident graph, with no CPU involvement in the data and
  control path
- Manages the full DOCA verbs lifecycle (CQ, QP, ring buffer, GPU export)

## Requirements

- Holoscan SDK 4.0.0 or later
- [NVIDIA DOCA SDK 3.2.1+](https://docs.nvidia.com/doca/sdk/doca-developer-guide/index.html) with GPUNetIO support
- Holoscan Sensor Bridge (HSB) 2.5.0 or later
- CUDA-capable GPU with GPUDirect RDMA support (Ampere or later, e.g. RTX A6000)
- aarch64 platform (e.g. IGX Orin)
- RoCE-capable NIC (e.g., ConnectX)
- libibverbs (InfiniBand Verbs library)

## Usage

### C++

```cpp
#include <hsb_roce_receiver_doca_gpunetio/doca_roce_receiver_op.hpp>

// In your application setup:
auto receiver = make_operator<hololink::operators::DocaRoceReceiverOp>(
    "doca_roce_receiver",
    holoscan::Arg("hololink_channel", data_channel),
    holoscan::Arg("device_start", start_fn),
    holoscan::Arg("device_stop", stop_fn),
    holoscan::Arg("frame_size", frame_size),
    holoscan::Arg("ibv_name", "roceP5p3s0f0"),
    holoscan::Arg("ibv_port", 1u),
    holoscan::Arg("pages", 2u)
);
```

## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `hololink_channel` | DataChannel* | - | Pointer to Hololink DataChannel object |
| `device_start` | std::function<void()> | - | Function called to start the sensor device |
| `device_stop` | std::function<void()> | - | Function called to stop the sensor device |
| `frame_size` | size_t | 0 | Size of one frame in bytes |
| `ibv_name` | string | "roceP5p3s0f0" | InfiniBand Verbs device name |
| `ibv_port` | uint32_t | 1 | Port number of IBV device |
| `pages` | uint32_t | 2 | Number of ring buffer pages |

## Build

This operator is built as part of the HoloHub build system when the `doca` mode
is selected:

```bash
./holohub build imx274_gpu_resident doca
```

Or enable the operator explicitly:

```bash
./holohub build imx274_gpu_resident --build-with hsb_roce_receiver_doca_gpunetio
```

## Dependencies

- `hololink::core` - Core hololink library
- `hololink::common` - Common hololink utilities
- `holoscan::core` - Holoscan SDK
- `DOCA SDK` - NVIDIA DOCA verbs, GPUNetIO, and common libraries
- `ibverbs` - InfiniBand Verbs library
- `CUDA` - CUDA driver and runtime APIs
- `fmt` - Formatting library
