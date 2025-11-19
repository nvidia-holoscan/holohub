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
Sends messages through a configured UcxxEndpoint. This operator serializes Holoscan messages and transmits them over the network using UCX.

**Parameters:**
- `endpoint`: Shared pointer to a UcxxEndpoint resource
- `tag`: Message tag for identifying message types (uint64_t)

### 3. **UcxxReceiverOp** (Operator)
Receives messages through a configured UcxxEndpoint. This operator listens for incoming messages, deserializes them, and outputs them to downstream operators.

**Parameters:**
- `endpoint`: Shared pointer to a UcxxEndpoint resource
- `tag`: Message tag for filtering received messages (uint64_t)
- `schema_name`: Name of the message schema for deserialization
- `buffer_size`: Size of the receive buffer

## Key Features

- **High Performance**: Leverages UCX for optimized network communication with support for RDMA and other high-speed interconnects
- **Low Latency**: Efficient zero-copy message transfers where possible
- **Flexible Topology**: Supports both client/server and peer-to-peer communication patterns
- **Message Serialization**: Uses FlatBuffers for efficient message serialization
- **Asynchronous Operations**: Non-blocking send and receive operations for better pipeline performance
- **Cross-Platform**: Supports both x86_64 and aarch64 architectures

## Use Cases

- Distributing Holoscan pipelines across multiple nodes
- Separating sensor acquisition from processing workloads
- Building multi-GPU processing pipelines with inter-node communication
- Creating scalable, distributed AI inference pipelines

## Requirements

- **Holoscan SDK**: Version 3.8.0 or higher
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
  Arg("schema_name", "MyMessageSchema"),
  Arg("buffer_size", 1024 * 1024)  // 1MB buffer
);
```

## Notes

- Ensure that the UCX library is properly installed and configured on your system
- Network connectivity must be established between nodes before communication can occur
- Message tags must match between sender and receiver pairs
- The endpoint should be initialized before the sender and receiver operators
- Consider firewall rules and network security when deploying distributed applications

