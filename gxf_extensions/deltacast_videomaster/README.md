# VideoMaster GXF Extension

This GXF extension provides support for VideoMaster SDK.

## Requirements

This extension requires the VideoMaster SDK

## Building the extension

As part of Holohub, running CMake on Holohub and point to Holoscan SDK install tree.

## Tests

| Application             | Device                  | Configuration                       | Holoscan SDK 2.8    |
| ----------------------- | ----------------------- | ----------------------------------- | ------------------- |
| deltacast_transmitter   | DELTA-12G-elp-key 11    | TX0 (SDI) / ~~RDMA~~                | PASSED              |
| deltacast_transmitter   | DELTA-12G-elp-key 11    | TX0 (SDI) / RDMA                    |                     |
| deltacast_transmitter   | DELTA-12G11-hmi11-e-key | TX0 (SDI) / ~~RDMA~~                | PASSED              |
| deltacast_transmitter   | DELTA-12G11-hmi11-e-key | TX0 (SDI) / RDMA                    |                     |
| deltacast_transmitter   | DELTA-12G11-hmi11-e-key | TX1 (HDMI) / ~~RDMA~~               | PASSED              |
| deltacast_transmitter   | DELTA-12G11-hmi11-e-key | TX1 (HDMI) / RDMA                   |                     |
| endoscopy_tool_tracking | DELTA-12G-elp-key 11    | RX0 (SDI) / ~~overlay~~ / ~~RDMA~~  | PASSED              |
| endoscopy_tool_tracking | DELTA-12G-elp-key 11    | RX0 (SDI) / ~~overlay~~ / RDMA      |                     |
| endoscopy_tool_tracking | DELTA-12G-elp-key 11    | RX0/TX0 (SDI) / overlay / ~~RDMA~~  |                     |
| endoscopy_tool_tracking | DELTA-12G-elp-key 11    | RX0/TX0 (SDI) / overlay / RDMA      |                     |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX0 (SDI) / ~~overlay~~ / ~~RDMA~~  |                     |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX0 (SDI) / ~~overlay~~ / RDMA      |                     |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX0/TX0 (SDI) / overlay / ~~RDMA~~  |                     |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX0/TX0 (SDI) / overlay / RDMA      |                     |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX1 (HDMI) / ~~overlay~~ / ~~RDMA~~ | PASSED              |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX1 (HDMI) / ~~overlay~~ / RDMA     |                     |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX1/TX1 (HDMI) / overlay / ~~RDMA~~ |                     |
| endoscopy_tool_tracking | DELTA-12G11-hmi11-e-key | RX1/TX1 (HDMI) / overlay / RDMA     |                     |