# Video Streaming Server

This application demonstrates how to use the Holoscan SDK to create a streaming server application that can send video streams to streaming clients.

## Requirements

- NVIDIA GPU
- CUDA 12.1 or higher
- Holoscan SDK 3.2.0 or higher

## Setup Instructions

**⚠️ Important**: Before building this application, you must first download the required streaming server binaries from NGC.

**📖 For detailed setup instructions, see**: [Streaming Server Operator Setup](../../operators/streaming_server/README.md#building-the-operator)

Quick summary:

```bash
cd <your_holohub_path>/operators/streaming_server 
ngc registry resource download-version "nvidia/holoscan_server_cloud_streaming:0.1"
unzip -o holoscan_server_cloud_streaming_v0.1/holoscan_server_cloud_streaming.zip

# Copy the appropriate architecture libraries to lib/ directory
# For x86_64 systems:
cp lib/x86_64/*.so* lib/
cp -r lib/x86_64/plugins lib/
# For aarch64 systems:
# cp lib/aarch64/* lib/

# Clean up architecture-specific directories and NGC download directory
rm -rf lib/x86_64 lib/aarch64
rm -rf holoscan_server_cloud_streaming_v0.1
```

## Running the Application

To run the application:

```bash
./holohub run video_streaming_server
```

### Command Line Options

- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_server_demo.yaml)
- `-d, --data <directory>`: Data directory (default: environment variable HOLOSCAN_INPUT_PATH or current directory)

## Configuration

The application can be configured using a YAML file. By default, it looks for `streaming_server_demo.yaml` in the current directory.

## Network Configuration

### Port Availability Check

Before setting up HAProxy or cloud functions, it's recommended to check if the required ports are available and not already in use. Use the provided port checking script:

```bash
# Check if the default streaming port (49010) is available
./check_port.sh 49010

# Check HAProxy port (if using custom port)
./check_port.sh 8080

# Check any specific port
./check_port.sh [PORT_NUMBER]
```

The script will show:

- ✅ **Port status**: Whether the port is listening or available
- 🔧 **Process information**: What processes are using the port (if any)
- 📋 **Port details**: Port type classification and availability for binding
- 🛠️ **Troubleshooting**: Helps identify port conflicts before deployment

**Common streaming ports to check:**

- `49010` - Default streaming server port
- `48010` - Alternative streaming port  
- `47999` - RTSP alternative port
- `8080` - Default HAProxy port

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

- [Video Streaming Client Application](../video_streaming_client/README.md)

### Operators

- [Streaming Server Operator](../../operators/streaming_server/README.md) - Detailed setup, configuration, and deployment instructions
- [Streaming Client Operator](../../operators/streaming_client/README.md) - Client-side streaming operator documentation
