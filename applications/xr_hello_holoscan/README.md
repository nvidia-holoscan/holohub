# XR "Hello Holoscan"

This application provides a simple scene demonstrating mixed reality viewing with Holoscan SDK.

## Background

We created this test application as part of a collaboration between the Magic Leap and NVIDIA Holoscan teams.
See the [`volume_rendering_xr`](/applications/volume_rendering_xr/) application for a demonstration of medical viewing
in XR with Holoscan SDK.

## Description

The application provides a blueprint for how to set up a mixed reality scene for viewing with Holoscan SDK and
HoloHub components.

The mixed reality demonstration scene includes:
- Static components such as scene axes and cube primitives;
- A primitive overlay on the tracked controller input;
- A static UI showcasing sensor inputs and tracking.

## Getting Started

Refer to the [`volume_rendering_xr` README](/applications/volume_rendering_xr/README.md#prerequisites) for details on hardware, firmware, and software prerequisites.

To run the application, run the following command in the HoloHub folder on your host machine:
```bash
dev_container build_and_run xr_hello_holoscan
```

To pair your Magic Leap 2 device with the host, open the QR Reader application in the ML2 headset and scan the QR code printed in console output on the host machine.
