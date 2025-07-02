# DDS Video: Real-time Video Streaming with RTI Connext & H.264

This application demonstrates how to encode video frames with H.264 using the multimedia 
extension over DDS.

The application can be run as either a publisher or as a subscriber. In either case,
it will use the [VideoFrame](../../../operators/dds/video/VideoFrame.idl) data topic
registered by the `DDSVideoPublisherOp` or `DDSVideoSubscriberOp` operators in order
to write or read the video frame data to/from the DDS databus, respectively.

When run as a publisher, the source for the input video frames can come from either an
attached V4L2-compatible camera via the `V4L2VideoCaptureOp` operator or a video file via the
`VideoStreamReplayerOp`. This can be configured in the `source` field inside the 
[dds_h264.yaml](./dds_h264.yaml) configuration file.

When run as a subscriber, the application will use Holoviz to render the received
video frames to the display. 

## Prerequisites

- This application requires [RTI Connext](https://content.rti.com/l/983311/2024-04-30/pz1wms)
be installed and configured with a valid RTI Connext license prior to use. 
- V4L2 capable device

> [!NOTE]
> Instructions below are based on the `.run' installer from RTI Connext. Refer to the
> [Linux installation](https://community.rti.com/static/documentation/developers/get-started/full-install.html)
> for details.


## Quick Start

```bash
# Start the publisher
./dev_container build_and_run dds_h264 --container_args "-v $HOME/rti_connext_dds-7.3.0:/opt/rti.com/rti_connext_dds-7.3.0/" --run_args "-p"

# Start the subscriber
./dev_container build_and_run dds_h264 --container_args "-v $HOME/rti_connext_dds-7.3.0:/opt/rti.com/rti_connext_dds-7.3.0/" --run_args "-s"
```


## Building the Application

To build on an IGX devkit (using the `armv8` architecture), follow the
[instructions to build Connext DDS applications for embedded Arm targets](https://community.rti.com/kb/how-do-i-create-connext-dds-application-rti-code-generator-and-build-it-my-embedded-target-arm)
up to, and including, step 5 (Installing Java and setting JREHOME).

To build the application, the `RTI_CONNEXT_DDS_DIR` CMake variable must point to
the installation path for RTI Connext. This can be done automatically by setting
the `NDDSHOME` environment variable to the RTI Connext installation directory
(such as when using the RTI `setenv` scripts), or manually at build time, e.g.:

```sh
$ ./run build dds_h264 --configure-args -DRTI_CONNEXT_DDS_DIR=~/rti/rti_connext_dds-7.3.0
```

### Building with a Container

Due to the license requirements of RTI Connext it is not currently supported to
install RTI Connext into a development container. Instead, Connext should be
installed onto the host as above and then the development container can be
launched with the RTI Connext folder mounted at runtime. To do so, ensure that
the `NDDSHOME` and `CONNEXTDDS_ARCH` environment variables are set (which can be
done using the RTI `setenv` script) and use the following:

```sh
# 1. Build the container
./dev_container build --docker_file applications/dds/dds_h264/Dockerfile
# 2. Launch the container
./dev_container launch --docker_opts "-v $HOME/rti_connext_dds-7.3.0:/opt/rti.com/rti_connext_dds-7.3.0/"
# 3. Build the application
./run build dds_h264
# Continue to the next section to run the application with the publisher. 
# Open a new terminal to repeat step #2 and launch a new container for the subscriber.
```



## Running the Application

Both a publisher and subscriber process must be launched to see the result of
writing to and reading the video stream from DDS, respectively.

To run the publisher process, use the `-p` option:

```sh
$ ./run launch dds_h264 --extra_args "-p"
```

To run the subscriber process, use the `-s` option:

```sh
$ ./run launch dds_h264 --extra_args "-s"
```

If running the application generates an error about `RTI Connext DDS No Source
for License information`, ensure that the RTI Connext license has either been
installed system-wide or the `NDDSHOME` environment variable has been set to
point to your user's RTI Connext installation path.

Note that these processes can be run on the same or different systems, so long as they
are both discoverable by the other via RTI Connext. If the processes are run on
different systems then they will communicate using UDPv4, for which optimizations have
been defined in the default `qos_profiles.xml` file. These optimizations include
increasing the buffer size used by RTI Connext for network sockets, and so the systems
running the application must also be configured to increase their maximum send and
receive socket buffer sizes. This can be done by running the `set_socket_buffer_sizes.sh`
script within this directory:

```sh
$ ./set_socket_buffer_sizes.sh
```

For more details, see the [RTI Connext Guide to Improve DDS Network Performance on Linux Systems](https://community.rti.com/howto/improve-rti-connext-dds-network-performance-linux-systems)

The QoS profiles used by the application can also be modified by editing the
`qos_profiles.xml` file in the application directory. For more information about modifying
the QoS profiles, see the [RTI Connext Basic QoS](https://community.rti.com/static/documentation/connext-dds/7.3.0/doc/manuals/connext_dds_professional/getting_started_guide/cpp11/intro_qos.html)
tutorial or the [RTI Connext QoS Reference Guide](https://community.rti.com/static/documentation/connext-dds/7.3.0/doc/manuals/connext_dds_professional/qos_reference/index.htm).

## Benchmarks

We collected latency benchmark results from the log output of the subscriber. The benchmark is conducted on x86_64 with NVIDIA ADA6000 GPU.

### Single System Setup

**Source**: Video Stream Replayer
**Resolution**: 854x480

| Configuration     | FPS     | AVg. Transfer Time | Jitter  | Input Size | Avg. Encoded Size |
|-------------------|---------|--------------------|---------|------------|-------------------|
| `realtime: false` | 685.068 | 1.576ms            | 0.840ms | 1,229,760  | 25,053            |
| `realtime: true`  | 30.049  | 0.150ms            | 0.059ms | 1,229,760  | 26,800            |

**Source**    : V4L2 Camera
**Frame Rate**: 30

| Resolution | FPS    | AVg. Transfer Time | Jitter  | Input Size | Avg. Encoded Size |
|------------|--------|--------------------|---------|------------|-------------------|
| 640x480    | 30.169 | 0.098ms            | 0.030ms | 921,600    | 16,176            |
| 1920x1080  | 30.281 | 0.104ms            | 0.040ms | 6,220,800  | 86,222            |

### Multiple System Setup

The two systems are connected via VPNin this scenario.

**Average Ping Latency**: 22.529ms


**Source**: Video Stream Replayer
**Resolution**: 854x480

| Configuration     | FPS     | AVg. Transfer Time | Jitter  | Input Size | Avg. Encoded Size |
|-------------------|---------|--------------------|---------|------------|-------------------|
| `realtime: false` | 607.581 | 12.278ms           | 3.595ms | 1,229,760  | 22,679            |
| `realtime: true`  | 30.050  | 12.937ms           | 3.856ms | 1,229,760  | 26,741            |


**Source**    : V4L2 Camera
**Frame Rate**: 30

| Resolution | FPS    | AVg. Transfer Time | Jitter  | Input Size | Avg. Encoded Size |
|------------|--------|--------------------|---------|------------|-------------------|
| 640x480    | 30.047 | 10.771ms           | 4.621ms | 921,600    | 11,571            |
| 1920x1080  | 28.877 | 14.322ms           | 3.420ms | 6,220,800  | 52,273            |
