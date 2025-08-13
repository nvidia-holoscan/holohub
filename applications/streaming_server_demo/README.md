# Streaming Server Demo

This application demonstrates how to use the Holoscan SDK to create a streaming server application that can send video streams to streaming clients.

## Requirements

- NVIDIA GPU
- CUDA 12.1 or higher
- Holoscan SDK 2.5.0 or higher

## Setup Instructions

**⚠️ Important**: Before building this application, you must first download the required streaming server binaries from NGC.

**📖 For detailed setup instructions, see**: [Streaming Server Operator Setup](../../operators/streaming_server/README.md#building-the-operator)

Quick summary:
```bash
# Download the Holoscan Server Cloud Streaming library from NGC
ngc registry resource download-version nvidia/holoscan_server_cloud_streaming:0.1
# Move to the operator's lib directory
mv holoscan_server_cloud_streaming operators/streaming_server/lib
```

## Building the Application

To build the application, run:

```bash
./holohub build streaming_server_demo
```

## Running the Application

To run the application:

```bash
./holohub run streaming_server_demo
```

### Command Line Options

- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_server_demo.yaml)
- `-d, --data <directory>`: Data directory (default: environment variable HOLOSCAN_INPUT_PATH or current directory)

## Configuration

The application can be configured using a YAML file. By default, it looks for `streaming_server_demo.yaml` in the current directory.

## Network Configuration

### HAProxy Requirements

For streaming clients to connect successfully, the following network requirements must be met:

- **HAProxy Accessibility**: HAProxy must be accessible by the client. The client does not need to be on the same network as HAProxy, but it must have a clear network path to reach HAProxy (i.e., the client must be able to "see" HAProxy, but HAProxy does not necessarily need to see the client).

- **Cross-Network Considerations**: If the client and server are on different networks, the streaming path needs to be handled accordingly:
  - The client must be able to reach the server's UDP ports
  - The opposite direction is not required since return traffic will be treated as responding traffic
  - Ensure proper firewall rules and NAT configurations allow this asymmetric communication

### Network Path Requirements

- Client → HAProxy: Required (client must be able to initiate connection)
- Client → Server UDP Ports: Required (for streaming data)
- Server → Client: Not required (handled as response traffic)

## Related Documentation

### Applications
- [Streaming Client Demo Application](../streaming_client_demo/README.md)

### Operators
- [Streaming Server Operator](../../operators/streaming_server/README.md) - Detailed setup, configuration, and deployment instructions
- [Streaming Client Operator](../../operators/streaming_client/README.md) - Client-side streaming operator documentation 