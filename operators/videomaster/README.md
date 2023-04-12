# VideoMaster GXF Operator

This library contains two operators:
- videomaster_source: get signal from capture card
- videomaster_transmitter: generate signal

These operators wrap the GXF extension to provide support for VideoMaster SDK.

## Requirements

This operator requires the VideoMaster SDK from Deltacast.

## Building the operator

As part of Holohub, running CMake on Holohub and point to Holoscan SDK install tree.

The path to the VideoMaster SDK is also mandatory and can be given through the VideoMaster_SDK_DIR parameter.
