# Streaming Server Operator

The `streaming_server` operator provides a streaming server implementation that can receive and send video frames to connected clients. It wraps the StreamingServer interface to provide seamless integration with Holoscan applications.

## `holoscan::ops::StreamingServerOp`

This operator class implements a streaming server that can:
- Accept incoming client connections
- Receive video frames from clients
- Send video frames to clients
- Handle multiple client connections (optional)
- Manage streaming events through callbacks

### Parameters

- **`width`**: Width of the video frames in pixels
  - type: `uint32_t`
  - default: 1920

- **`height`**: Height of the video frames in pixels
  - type: `uint32_t`
  - default: 1080

- **`fps`**: Frame rate of the video
  - type: `uint32_t`
  - default: 30

- **`port`**: Port used for streaming server
  - type: `uint16_t`
  - default: 8080

- **`multi_instance`**: Allow multiple server instances
  - type: `bool`
  - default: false

- **`server_name`**: Name identifier for the server
  - type: `std::string`
  - default: "StreamingServer"

- **`receive_frames`**: Whether to receive frames from clients
  - type: `bool`
  - default: true

- **`send_frames`**: Whether to send frames to clients
  - type: `bool`
  - default: false

- **`allocator`**: Memory allocator for frame data
  - type: `std::shared_ptr<Allocator>`

### Input Ports

- **`input_frames`**: Input port for frames to be sent to clients
  - type: `holoscan::gxf::Entity`

### Output Ports

- **`output_frames`**: Output port for frames received from clients
  - type: `holoscan::gxf::Entity`

### Example Usage

```cpp
// Create the operator with configuration
auto streaming_server = make_operator<ops::StreamingServerOp>(
    "streaming_server",
    Arg("width") = 1920,
    Arg("height") = 1080,
    Arg("fps") = 30,
    Arg("port") = 8080,
    Arg("multi_instance") = false,
    Arg("server_name") = "MyStreamingServer",
    Arg("receive_frames") = true,
    Arg("send_frames") = true,
    Arg("allocator") = make_resource<UnboundedAllocator>("pool")
);

// Add it to your graph
graph.add(streaming_server);

// Connect it to other operators
graph.connect(source, "output", streaming_server, "input_frames");
graph.connect(streaming_server, "output_frames", sink, "input");
``` 

## Building the operator

In order to build the server operator, you must first download the server binaries form NGC and add to the `lib` directory in the `streaming_server` operator folder

Download the Holoscan Server Cloud Streaming library from NGC:
https://catalog.ngc.nvidia.com/orgs/nvidia/resources/holoscan_server_cloud_streaming

```bash
# Download and extract the library
ngc registry resource download-version nvidia/holoscan_server_cloud_streaming:0.1
# Move the extracted files to the lib directory
mv holoscan_server_cloud_streaming lib
```

### Deployment on NVCF

The Holoscan cloud steaming stack provides plugins with endpoints required to deploy the server docker container as a streaming function.
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

## Supported Platforms

- Linux x86_64
- NVCF Cloud instances 

For more information on NVCF Cloud functions, please refer to [NVIDIA Cloud Functions documentation](https://docs.nvidia.com/cloud-functions/user-guide/latest/cloud-function/function-creation.html#function-creation).

