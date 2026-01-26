# UCXX Operators

## Overview

The UCXX operators provide high-performance, low-latency communication capabilities for Holoscan applications using the Unified Communication X (UCX) framework. These operators enable efficient data transfer between distributed Holoscan applications, making them ideal for multi-node deployments and distributed processing pipelines.

## Components

This components group includes three key components:

### 1. **UcxxEndpoint** (Resource)
A Holoscan Resource that manages a UCXX endpoint for UCX communication. It handles connection establishment, either by listening for incoming connections or by connecting to a remote endpoint.

**Parameters:**
- `hostname`: The hostname or IP address for the connection
- `port`: The port number to listen on or connect to
- `listen`: Boolean flag indicating whether to listen for connections (server mode) or connect to a remote endpoint (client mode)

### 2. **UcxxSenderOp** (Operator)
Sends tensor messages through a configured `UcxxEndpoint` using UCXX/UCX. The sender uses a two-phase protocol:
1) send a small CPU header containing tensor metadata, then
2) send the tensor payload from the tensor’s underlying pointer (CPU or GPU).

**Parameters:**
- `endpoint`: Shared pointer to a `UcxxEndpoint` resource
- `tag`: Base message tag for identifying message types (`uint64_t`). Note: this operator consumes two tags: `tag` (header) and `tag+1` (payload).
- `blocking`: If true, the operator does not execute until the endpoint is connected. If false (default), it drains inputs and drops sends while disconnected.
- `max_in_flight`: Maximum number of in-flight async send requests to retain (default: 1). When exceeded, new inputs are dropped to bound memory retention if the network/receiver stalls.

**Async send lifetime and backpressure behavior:**
- Sends are asynchronous. Any buffers passed to UCX must remain valid until the corresponding UCX request completes.
- The sender retains a keepalive handle to the input entity (and any temporary tensor wrapper) until both header and payload requests complete, preventing pooled buffers from being recycled while UCX is still reading them.
- On disconnect, the sender requests cancellation of any in-flight sends but retains keepalive state until UCX reports completion. While disconnected, new inputs are dropped when `blocking` is false.

**Zero-copy and transport selection (UCX-managed):**
- The operator itself does not copy the payload into a staging buffer; it hands UCX the original CPU/GPU pointer. Whether the transfer is truly “zero-copy” end-to-end depends on UCX’s selected protocol and transports.
- UCX may choose eager vs rendezvous and may use GPU-aware transports when available (for example, same-node CUDA IPC, or GPUDirect RDMA on capable systems), but it may also internally stage/copy depending on configuration, message size, and transport support.

### 3. **UcxxReceiverOp** (Operator)
Receives messages through a configured UcxxEndpoint. This operator listens for incoming messages, deserializes them, and outputs them to downstream operators.

**Parameters:**
- `endpoint`: Shared pointer to a `UcxxEndpoint` resource
- `tag`: Base message tag for filtering received messages (`uint64_t`). Note: this operator consumes two tags: `tag` (header) and `tag+1` (payload).
- `buffer_size`: Tensor payload buffer size in bytes (required)
- `receive_on_device`: Allocate the payload buffer on device (GPU) if true, host (CPU) if false (default: true)
- `allocator`: Allocator used for the receive buffer allocation

**Async receive behavior:**
- The receiver posts two receives in parallel: one for the CPU header (tensor metadata) and one for the tensor payload.
- The receiver allocates a payload buffer (GPU or CPU) and receives into it; it then wraps that buffer into an output tensor and releases it when downstream is done.

## Key Features

- **High Performance**: Leverages UCX for optimized network communication. UCX also supports Direct Memory Access with
  RDMA, Infiniband, etc.
- **Low Latency**: Efficient zero-copy message transfers where possible
- **Flexible Topology**: Supports both client/server and peer-to-peer communication patterns
- **Message Serialization**: Uses tensor serialization based on the NVIDIA GXF/Holoscan serialization approach
  for efficient message serialization
- **Asynchronous Operations**: Non-blocking send and receive operations for better pipeline performance
- **Cross-Platform**: Supports both x86_64 and aarch64 architectures

## Use Cases

- Distributing Holoscan pipelines across multiple nodes
- Separating sensor acquisition from processing workloads
- Building multi-GPU processing pipelines with inter-node communication
- Creating scalable, distributed AI inference pipelines

## Requirements

- **Holoscan SDK**: Version 3.9.0 or higher
- **UCXX Library**: UCX C++ bindings
- **Platforms**: x86_64, aarch64
- **Dependencies**: UCX (Unified Communication X) framework

## Example Configuration

```cpp
// Create endpoint resource (server mode)
auto endpoint = make_resource<UcxxEndpoint>(
  "ucxx_endpoint",
  Arg("hostname", "0.0.0.0"),
  Arg("port", 12345),
  Arg("listen", true)
);

// Sender operator
auto sender = make_operator<UcxxSenderOp>(
  "sender",
  Arg("endpoint", endpoint),
  Arg("tag", 1UL)
);

// Receiver operator
auto receiver = make_operator<UcxxReceiverOp>(
  "receiver",
  Arg("endpoint", endpoint),
  Arg("tag", 1UL),
  Arg("buffer_size", 1024 * 1024)  // 1MB buffer
);
```

## Notes

- Ensure that the UCX library is properly installed and configured on your system
- Network connectivity must be established between nodes before communication can occur
- Message tags must match between sender and receiver pairs
- The endpoint should be initialized before the sender and receiver operators
- Consider firewall rules and network security when deploying distributed applications
