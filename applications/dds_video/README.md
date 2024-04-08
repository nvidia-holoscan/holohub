# DDS Video Application

The DDS Video application demonstrates how video frames can be written to or
read from a DDS databus in order to provide flexible integration between
Holoscan applications and other applications (using Holoscan or not) via DDS.

The application can be run as either a publisher or as a subscriber. In either case,
it will use the [VideoFrame](../../operators/dds/video/VideoFrame.idl) data topic
registered by the `DDSVideoPublisherOp` or `DDSVideoSubscriberOp` operators in order
to write or read the video frame data to/from the DDS databus, respectively.

When run as a publisher, the source for the input video frames will come from an
attached V4L2-compatible camera via the `V4L2VideoCaptureOp` operator.

When run as a subscriber, the application will use Holoviz to render the received
video frames to the display. In addition to the video stream, the subscriber
application will also subscribe to the `Square`, `Circle`, and `Triangle` topics
as used by the [RTI Shapes Demo](https://www.rti.com/free-trial/shapes-demo).
Any shapes received by this subscriber will also be overlayed on top of the
Holoviz output.

![DDS Video Application Workflow](docs/workflow_dds_video_app.png)

## Building the Application

This application requires [RTI Connext](https://www.rti.com/products) be
installed and configured with a valid RTI Connext license prior to use.

To build the application, the `RTI_CONNEXT_DDS_DIR` variable must be provided
to specify the installation path for RTI Connext. For example,

```sh
$ ./run build dds_video --configure-args -DRTI_CONNEXT_DDS_DIR=~/rti/rti_connext_dds-6.1.2
```

## Running the Application

Both a publisher and subscriber process must be launched to see the result of
writing to and reading the video stream from DDS, respectively.

To run the publisher process, use the `-p` option:

```sh
$ ./dds_video -p
```

To run the subscriber process, use the `-s` option:

```sh
$ ./dds_video -s
```

Note that these processes can be run on the same or different systems, so long as they
are both discoverable by the other via RTI Connext.

The QoS profiles used by the application can also be modified by editing the
`qos_profiles.xml` file in the application directory. For more information about modifying
the QoS profiles, see the [RTI Connext Basic QoS](https://community.rti.com/static/documentation/connext-dds/6.0.1/doc/manuals/connext_dds/getting_started/cpp11/intro_qos.html#)
tutorial or the [RTI Connext QoS Reference Guide](https://community.rti.com/static/documentation/connext-dds/6.0.1/doc/manuals/connext_dds/RTI_ConnextDDS_CoreLibraries_QoS_Reference_Guide.pdf).
