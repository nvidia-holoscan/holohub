# Streaming Client Demo

This application demonstrates how to use the Holoscan SDK to create a streaming client application that can send and receive video streams to and from a streaming server.

## Requirements

- NVIDIA GPU
- CUDA 12.1 or higher
- Holoscan SDK 3.4.0 or higher

## Dependencies

In order to build the client operator, you must first download the client binaries from NGC:

```bash
# Download using NGC CLI
cd <your_holohub_path>/operators/streaming_client_enhanced
ngc registry resource download-version "nvidia/holoscan_client_cloud_streaming:0.2"
unzip -o holoscan_client_cloud_streaming_v0.2/holoscan_client_cloud_streaming.zip -d holoscan_client_cloud_streaming

# Clean up extraction directory and NGC download directory
rm -rf streaming_client_enhanced holoscan_client_cloud_streaming_v0.2
```

## Running the Application

To run the application use this command from the holohub root directory:

```bash
./holohub run --docker-opts='-e EnableHybridMode=1 -e device=/dev/video0' streaming_client_demo_enhanced --language cpp --config streaming_client_demo.yaml
```

**Important Notes:**
- The application is already configured to use V4L2 camera input (see `source: "v4l2"` in the YAML config)
- Make sure your camera is connected and accessible at `/dev/video0`
- The `-e device=/dev/video0` environment variable ensures the camera device is available inside the Docker container
- The `--config applications/streaming_client_demo_enhanced/cpp/streaming_client_demo.yaml` specifies the full path to the configuration file

### Command Line Options

- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_client_demo.yaml)
- `-d, --data <directory>`: Data directory (default: environment variable HOLOSCAN_INPUT_PATH or current directory)

### Troubleshooting Camera Issues

If you're getting errors about missing video files, it means the application might be defaulting to replayer mode instead of V4L2. Check these things:

1. **Verify your camera is detected:**
   ```bash
   ls -la /dev/video*
   v4l2-ctl --device=/dev/video0 --info
   ```

2. **Test camera with the recommended resolution:**
   ```bash
   v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=YUYV --stream-mmap --stream-count=10
   ```

3. **If you need to use video file instead of camera**, modify the config to use replayer:
   ```bash
   # Edit applications/streaming_client_demo_enhanced/cpp/streaming_client_demo.yaml
   # Change: source: "v4l2"
   # To:     source: "replayer"
   ```

## Configuration

The application can be configured using a YAML file. By default, it looks for `streaming_client_demo.yaml` in the current directory. The configuration is set to use V4L2 camera input by default (`source: "v4l2"`).

## Expected Behavior and Logs

During normal operation, you may see the following messages in the logs, which are **expected and do not indicate errors**:

```
[warning] [double_buffer_transmitter.cpp:79] Push failed on 'output_frames'
[error] [gxf_io_context.cpp:673] Failed to publish output message with error: GXF_EXCEEDING_PREALLOCATED_SIZE
[error] [streaming_client.cpp:1299] Error in receive_frames processing: Failed to publish output message with error: GXF_EXCEEDING_PREALLOCATED_SIZE
[warning] [double_buffer_receiver.cpp:80] Push failed on 'receivers:0'
```

These messages occur due to buffer management during high-throughput streaming and frame format conversion (BGR to BGRA). The streaming client continues to operate normally despite these buffer warnings, as the system automatically handles buffer overflow scenarios by dropping frames when necessary. This is the intended behavior for real-time streaming applications to maintain performance.

## Operator Documentation

For detailed information about the underlying streaming client operator used in this application, see:

📋 **[Streaming Client Enhanced Operator](../../operators/streaming_client_enhanced/README.md)** - Complete operator documentation

The operator documentation includes:
- **Architecture and Implementation**: StreamingClientOp configuration and usage
- **Parameters and Configuration**: Detailed parameter descriptions and camera setup
- **Testing Documentation**: Comprehensive test suite with 25/25 tests passing
- **API Reference**: Complete API documentation and troubleshooting guides

## Related Applications

- **[Streaming Server Demo Enhanced](../streaming_server_demo_enhanced/README.md)** - Companion server application for bidirectional streaming 