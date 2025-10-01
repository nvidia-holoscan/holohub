# Streaming Server Operators

The `streaming_server_enhanced` operator provides a modular streaming server implementation with separate upstream, downstream, and resource components. This split architecture allows for better separation of concerns and more flexible streaming pipeline configurations.

## Architecture Components

### `holoscan::ops::StreamingServerResource`

A shared resource that manages the underlying StreamingServer instance and provides:
- Centralized server lifecycle management
- Event handling and callback management
- Configuration management for server parameters
- Frame sending and receiving coordination between operators

### `holoscan::ops::StreamingServerUpstreamOp`

An operator that receives video frames from streaming clients and outputs them as Holoscan tensors:
- Receives frames from connected clients via the StreamingServerResource
- Converts received frames to `holoscan::Tensor` format
- Provides duplicate frame detection to ensure unique frame processing
- Outputs tensors to the Holoscan processing pipeline

### `holoscan::ops::StreamingServerDownstreamOp`

An operator that receives Holoscan tensors and sends them to streaming clients:
- Takes `holoscan::Tensor` input from the processing pipeline
- Converts tensors back to video frame format
- Sends processed frames to connected clients via the StreamingServerResource
- Supports optional frame processing (mirroring, rotation, etc.)

## Benefits of Split Architecture

- **Modularity**: Each component has a single responsibility (resource management, receiving, or sending)
- **Flexibility**: You can use only upstream, only downstream, or both depending on your pipeline needs
- **Shared Resource**: Multiple operators can share the same StreamingServerResource instance
- **Better Testing**: Each component can be tested independently
- **Clear Data Flow**: Explicit tensor-based input/output ports make data flow obvious
- **Processing Integration**: Seamless integration with Holoscan's tensor processing pipeline

## Parameters

### StreamingServerResource Parameters

- **`port`**: Port used for streaming server
  - type: `uint16_t`
  - default: 48010

- **`is_multi_instance`**: Allow multiple server instances
  - type: `bool`
  - default: false

- **`server_name`**: Name identifier for the server
  - type: `std::string`
  - default: "HoloscanStreamingServer"

- **`width`**: Width of the video frames in pixels
  - type: `uint32_t`
  - default: 854

- **`height`**: Height of the video frames in pixels
  - type: `uint32_t`
  - default: 480

- **`fps`**: Frame rate of the video
  - type: `uint16_t`
  - default: 30

- **`enable_upstream`**: Enable upstream (receiving) functionality
  - type: `bool`
  - default: true

- **`enable_downstream`**: Enable downstream (sending) functionality
  - type: `bool`
  - default: true

### StreamingServerUpstreamOp Parameters

- **`width`**: Frame width (inherits from resource if not specified)
  - type: `uint32_t`
  - default: 854

- **`height`**: Frame height (inherits from resource if not specified)
  - type: `uint32_t`
  - default: 480

- **`fps`**: Frame rate (inherits from resource if not specified)
  - type: `uint32_t`
  - default: 30

- **`allocator`**: Memory allocator for tensor data
  - type: `std::shared_ptr<Allocator>`

- **`streaming_server_resource`**: Reference to StreamingServerResource
  - type: `std::shared_ptr<StreamingServerResource>`

### StreamingServerDownstreamOp Parameters

- **`width`**: Frame width (inherits from resource if not specified)
  - type: `uint32_t`
  - default: 854

- **`height`**: Frame height (inherits from resource if not specified)
  - type: `uint32_t`
  - default: 480

- **`fps`**: Frame rate (inherits from resource if not specified)
  - type: `uint32_t`
  - default: 30

- **`enable_processing`**: Enable frame processing (mirroring, etc.)
  - type: `bool`
  - default: false

- **`processing_type`**: Type of processing to apply
  - type: `std::string`
  - default: "none"
  - options: "none", "mirror", "rotate"

- **`allocator`**: Memory allocator for tensor data
  - type: `std::shared_ptr<Allocator>`

- **`streaming_server_resource`**: Reference to StreamingServerResource
  - type: `std::shared_ptr<StreamingServerResource>`

## Input/Output Ports

### StreamingServerUpstreamOp Ports

**Output Ports:**
- **`output_frames`**: Output port for frames received from clients
  - type: `holoscan::Tensor`

### StreamingServerDownstreamOp Ports

**Input Ports:**
- **`input_frames`**: Input port for frames to be sent to clients
  - type: `holoscan::Tensor`

## Example Usage

### Complete Pipeline Setup

```cpp
// Create allocator resource
auto allocator = make_resource<UnboundedAllocator>("allocator");

// Create the shared StreamingServerResource
auto streaming_server_resource = make_resource<ops::StreamingServerResource>(
    "streaming_server_resource",
    Arg("port") = 48010,
    Arg("server_name") = "MyStreamingServer",
    Arg("width") = 854,
    Arg("height") = 480,
    Arg("fps") = 30,
    Arg("enable_upstream") = true,
    Arg("enable_downstream") = true,
    Arg("is_multi_instance") = false
);

// Create upstream operator (receives frames from clients)
auto upstream_op = make_operator<ops::StreamingServerUpstreamOp>(
    "streaming_upstream",
    Arg("allocator") = allocator,
    Arg("streaming_server_resource") = streaming_server_resource
);

// Create downstream operator (sends frames to clients)
auto downstream_op = make_operator<ops::StreamingServerDownstreamOp>(
    "streaming_downstream",
    Arg("enable_processing") = false,
    Arg("processing_type") = "none",
    Arg("allocator") = allocator,
    Arg("streaming_server_resource") = streaming_server_resource
);


```
                                                 

## Building the operator

In order to build the server operator, you must first download the server binaries from NGC:

```bash
# Download using NGC CLI

cd <your_holohub_path>/operators/streaming_server_enhanced
ngc registry resource download-version "nvidia/holoscan_server_cloud_streaming:0.2"
unzip -o holoscan_server_cloud_streaming_v0.2/holoscan_server_cloud_streaming.zip -d holoscan_server_cloud_streaming

# Clean up extraction directory and NGC download directory
rm -rf streaming_server_enhanced holoscan_server_cloud_streaming_v0.2
```

### Deployment on NVCF

The Holoscan cloud streaming stack provides plugins with endpoints required to deploy the server docker container as a streaming function.
You can push the container and create/update/deploy the streaming function from the [web portal](https://nvcf.ngc.nvidia.com/functions).

#### Push Container

Note: You first must docker login to the NGC Container Registry before you can push containers to it:
https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#accessing-ngc-registry
Tag the container and push it to the container registry:

```bash
docker tag simple-streamer:latest {registry}/{org-id}/{container-name}:{version}
docker push {registry}/{org-id}/{container-name}:{version}
```

For example, if your organization name/id is 0494839893562652 and you want to push a container to the prod container registry
using the name my-simple-streamer at version 0.1.0 then run:

```bash
docker tag simple-streamer:latest nvcr.io/0494839893562652/my-simple-streamer:0.1.0
docker push nvcr.io/0494839893562652/my-simple-streamer:0.1.0
```

#### Set Variables

All the helper scripts below depend on the following environment variables being set:

```bash
# Required variables
export NGC_PERSONAL_API_KEY=<get from https://nvcf.ngc.nvidia.com/functions -> Generate Personal API Key>
export STREAMING_CONTAINER_IMAGE=<registry>/<org-id>/<container-name>:<version>
export STREAMING_FUNCTION_NAME=<my-simple-streamer-function-name>

# Optional variables (shown with default values)
export NGC_DOMAIN=api.ngc.nvidia.com
export NVCF_SERVER=grpc.nvcf.nvidia.com
export STREAMING_SERVER_PORT=49100
export HTTP_SERVER_PORT=8011
```

#### Create the Cloud Streaming Function

Create the streaming function by running the provided script after setting all the required variables:
```bash
./nvcf/create_streaming_function.sh
```

Once the function is created, export the `FUNCTION_ID` as a variable:

```bash
export STREAMING_FUNCTION_ID={my-simple-streamer-function-id}
```

#### Update Function

Update an existing streaming function by running the provided script after setting all the required variables:
```bash
./nvcf/update_streaming_function.sh
```

#### Deploy Function

Deploy the streaming function from the web portal: https://nvcf.ngc.nvidia.com/functions

#### Test Function

Start the test intermediate haproxy by running the provided script after setting all the required variables:

```bash
./nvcf/start_test_intermediate_haproxy.sh
```

Please note that the test haproxy server should be running on a separate machine, either on the client machine or a separate one.

Note: If the test haproxy is still running, and you wish to test the executable or docker file again you must first stop it:

```bash
./nvcf/stop_test_intermediate_haproxy.sh
```

## Testing

Testing is handled at the application level through the unified `video_streaming_demo_enhanced` integration test, which provides comprehensive end-to-end validation of the streaming server working with the client.

## Related Applications

- **[Streaming Server Demo Enhanced](../../../applications/video_streaming_demo_enhanced/video_streaming_demo_server/README.md)** - Complete application demonstrating the streaming server operators
- **[Streaming Client Demo Enhanced](../../../applications/video_streaming_demo_enhanced/video_streaming_demo_client/README.md)** - Companion client application for bidirectional streaming

## Supported Platforms

- Linux x86_64
- NVCF Cloud instances 

For more information on NVCF Cloud functions, please refer to [NVIDIA Cloud Functions documentation](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/function-creation.html#function-creation).

