# Streaming Client Demo Application

This application demonstrates how to use the Holoscan SDK to create a streaming client application that can receive video streams from a streaming server.

## Requirements

- NVIDIA GPU
- CUDA 12.1 or higher
- Holoscan SDK 2.5.0 or higher


## Building the Application

To build the application, run:

```bash
./holohub build streaming_client_demo
```

## Running the Application

To run the application:

```bash
./holohub run streaming_client_demo
```

### Command Line Options

- `-h, --help`: Show help message
- `-c, --config <file>`: Configuration file path (default: streaming_client_demo.yaml)
- `-d, --data <directory>`: Data directory (default: environment variable HOLOSCAN_INPUT_PATH or current directory)

## Configuration

The application can be configured using a YAML file. By default, it looks for `streaming_client_demo.yaml` in the current directory.

## Related Applications

- [Streaming Server Demo Application](../streaming_server_demo/README.md) 