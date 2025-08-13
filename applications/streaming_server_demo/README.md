# Streaming Server Demo

This application demonstrates how to use the Holoscan SDK to create a streaming server application that can send video streams to streaming clients.

## Requirements

- NVIDIA GPU
- CUDA 12.1 or higher
- Holoscan SDK 2.5.0 or higher


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

## Related Applications

- [Streaming Client Demo Application](../streaming_client_demo/README.md) 