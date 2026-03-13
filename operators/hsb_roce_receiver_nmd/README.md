# HSB RoCE Receiver Operator with No Host Metadata

This operator receives video frames over RoCE (RDMA over Converged Ethernet)
directly to GPU memory for high-performance, low-latency video ingestion.
# HSB RoCE Receiver Operator with No Host Metadata

This operator receives video frames over RoCE (RDMA over Converged Ethernet)
directly to GPU memory for high-performance, low-latency video ingestion.
It has an option to skip metadata processing on the host, so that metadata is never copied from the device to the host CPU. This is useful for Holoscan SDK GPU-resident applications.

This operator forks the [Holoscan Sensor Bridge 2.5.0 RoCE Receiver Operator](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/2.5.0-PB6/src/hololink/operators/roce_receiver/roce_receiver_op.hpp) implementation and extends it with `frame_memory_base` access to support GPU Resident operations.

## Overview

The `RoceReceiverOp` is a Holoscan operator that:

- Receives video frames via RDMA directly to GPU memory
- Supports multi-page buffering for continuous streaming
- Provides optional GPU-resident metadata processing
- Extends `BaseReceiverOp` from the hololink library

## Requirements

- Holoscan SDK 4.0.0 or later
- Holoscan Sensor Bridge ("Hololink") library installed and available
- CUDA-capable GPU with GPUDirect RDMA support
- libibverbs (InfiniBand Verbs library)
- RoCE-capable NIC (e.g., ConnectX)

## Usage

### C++

```cpp
#include <hsb_roce_receiver_nmd/roce_receiver_op.hpp>

// In your application setup:
auto receiver = make_operator<hololink::operators::RoceReceiverOp>(
    "roce_receiver",
    holoscan::Arg("hololink_channel", data_channel),
    holoscan::Arg("device_start", start_fn),
    holoscan::Arg("device_stop", stop_fn),
    holoscan::Arg("frame_context", cuda_context),
    holoscan::Arg("frame_size", frame_size),
    holoscan::Arg("ibv_name", "roceP5p3s0f0"),
    holoscan::Arg("ibv_port", 1u),
    holoscan::Arg("pages", 2u),
    holoscan::Arg("queue_size", 1u),
    holoscan::Arg("skip_host_metadata", false)
);

// Add to your pipeline
add_flow(receiver, next_operator, {{"output", "input"}});
```

## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `hololink_channel` | DataChannel* | - | Pointer to Hololink DataChannel object |
| `device_start` | std::function<void()> | - | Function called to start the device |
| `device_stop` | std::function<void()> | - | Function called to stop the device |
| `frame_context` | CUcontext | - | CUDA context for frame memory |
| `frame_size` | size_t | - | Size of one frame in bytes |
| `ibv_name` | string | "roceP5p3s0f0" | InfiniBand Verbs device name |
| `ibv_port` | uint32_t | 1 | Port number of IBV device |
| `pages` | uint32_t | 2 | Number of pages for receiver memory |
| `queue_size` | uint32_t | 1 | Number of buffers that can be queued |
| `skip_host_metadata` | bool | false | Skip copying metadata to host |
| `trim` | bool | false | Trim output to bytes_written |
| `use_frame_ready_condition` | bool | true | Use async condition for frame readiness |

## Output Metadata

The operator publishes the following metadata with each frame:

- `received_frame_number`: Sequence number of received frame
- `rx_write_requests`: Total RDMA write requests
- `received_s`, `received_ns`: Receive timestamp
- `imm_data`: Immediate data from RDMA write
- `frame_memory`: GPU device pointer to frame data
- `dropped`: Number of dropped frames
- `frame_number`: Frame number from device
- `timestamp_s`, `timestamp_ns`: Device timestamp
- `crc`: Frame CRC value
- `bytes_written`: Actual bytes written

## Variants

### RoceReceiverNoHostMetadata

A variant that keeps metadata on the GPU and avoids host copies. Enable this with `skip_host_metadata=true` for GPU-resident processing pipelines.

## Build

This operator is built as part of the HoloHub build system:

```bash
./run build hsb_roce_receiver_nmd
```

## Dependencies

- `hololink::core` - Core hololink library
- `hololink::common` - Common hololink utilities
- `hololink::operators::base_receiver` - Base receiver operator
- `holoscan::core` - Holoscan SDK
- `ibverbs` - InfiniBand Verbs library
- `CUDA` - CUDA driver API
- `fmt` - Formatting library
