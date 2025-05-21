# DELTACAST VideoMaster Operators

The DELTACAST VideoMaster operator provides functionality to capture and stream high-quality video streams from DELTACAST cards. It supports both SDI and HDMI input and output sources, enabling professional video capture in various formats and resolutions. DELTACAST VideoMaster operators are designed to work seamlessly with DELTACAST's hardware capabilities.

This library contains two operators:
- videomaster_source: Captures a signal from the DELTACAST capture card.
- videomaster_transmitter: Streams a signal through the DELTACAST capture card.

These operators wrap the GXF extension to provide support for the VideoMaster SDK.

## Requirements

* VideoMaster SDK: Operators require the VideoMaster SDK from Deltacast.
* DELTACAST Hardware: Compatible DELTACAST capture cards.
* VideoMaster driver: To detect and use DELTACAST capture cards.

## Parameters

### videomaster_source
The following parameters can be configured for this operator:

| Parameter | Type     | Description                                                                                          | Default |
| --------- | -------- | ---------------------------------------------------------------------------------------------------- | ------- |
| `board`   | uint32_t | Index of the DELTACAST.TV board to use as source                                                     | 0       |
| `rdma`    | bool     | Enable RDMA for video input (DELTACAST driver must be compiled with RDMA enabled to use this option) | false   |
| `input`   | uint32_t | Index of the RX channel to use on the selected board                                                 | 0       |

### videomaster_transmitter
The following parameters can be configured for this operator:

| Parameter        | Type     | Description                                                                                          | Default |
| ---------------- | -------- | ---------------------------------------------------------------------------------------------------- | ------- |
| `board`          | uint32_t | Index of the DELTACAST.TV board to use as source                                                     | 0       |
| `rdma`           | bool     | Enable RDMA for video input (DELTACAST driver must be compiled with RDMA enabled to use this option) | false   |
| `output`         | uint32_t | Index of the TX channel to use on the selected board                                                 | 0       |
| `width`          | uint32_t | The width of the output stream                                                                       | 1920    |
| `height`         | uint32_t | The height of the output stream                                                                      | 1080    |
| `progressive`    | bool     | interleaved or progressive                                                                           | true    |
| `framerate`      | uint32_t | The framerate of the output stream                                                                   | 60      |
| `enable_overlay` | bool     | Is overlay is add by card or not                                                                     | false   |

## Building the operator

As part of Holohub, running CMake on Holohub and point to Holoscan SDK install tree.

The path to the VideoMaster SDK is also mandatory and can be given through the VideoMaster_SDK_DIR parameter.

## Tests

All tests performed with the DELTACAST VideoMaster SDK `6.30`.

| Application             | Device                  | Configuration                       | Holoscan SDK 2.9 | Holoscan SDK 3.0 | Holoscan SDK 3.1 |
| ----------------------- | ----------------------- | ----------------------------------- | ---------------- | ---------------- | ---------------- |
| deltacast_transmitter   | DELTA-12G-elp-key 11    | TX0 (SDI) / ~~RDMA~~                | PASSED           | PASSED           | PASSED           |
| deltacast_transmitter   | DELTA-12G-elp-key 11    | TX0 (SDI) / RDMA                    | PASSED           | PASSED           | PASSED           |
| deltacast_transmitter   | DELTA-12G11-hmi11-e-key | TX0 (SDI) / ~~RDMA~~                | PASSED           | PASSED           | PASSED           |
| deltacast_transmitter   | DELTA-12G11-hmi11-e-key | TX0 (SDI) / RDMA                    | PASSED           | PASSED           | PASSED           |
| deltacast_transmitter   | DELTA-12G11-hmi11-e-key | TX1 (HDMI) / ~~RDMA~~               | PASSED           | PASSED           | PASSED           |
| deltacast_transmitter   | DELTA-12G11-hmi11-e-key | TX1 (HDMI) / RDMA                   | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G-elp-key 11    | RX0 (SDI) / ~~overlay~~ / ~~RDMA~~  | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G-elp-key 11    | RX0 (SDI) / ~~overlay~~ / RDMA      | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G-elp-key 11    | RX0/TX0 (SDI) / overlay / ~~RDMA~~  | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G-elp-key 11    | RX0/TX0 (SDI) / overlay / RDMA      | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX0 (SDI) / ~~overlay~~ / ~~RDMA~~  | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX0 (SDI) / ~~overlay~~ / RDMA      | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX0/TX0 (SDI) / overlay / ~~RDMA~~  | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX0/TX0 (SDI) / overlay / RDMA      | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX1 (HDMI) / ~~overlay~~ / ~~RDMA~~ | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX1 (HDMI) / ~~overlay~~ / RDMA     | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX1/TX1 (HDMI) / overlay / ~~RDMA~~ | PASSED           | PASSED           | PASSED           |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX1/TX1 (HDMI) / overlay / RDMA     | PASSED           | PASSED           | PASSED           |