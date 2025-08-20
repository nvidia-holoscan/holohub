# Deltacast Videomaster Receiver

This application demonstrates the use of videomaster_source to receive and display video streams from a Deltacast capture card using Holoviz for visualization.

## Overview

The Deltacast Receiver application provides a simplified pipeline for receiving video streams from Deltacast hardware and displaying them without AI processing or overlay components. It consists of:

- **VideoMasterSource**: Receives video streams from Deltacast capture card
- **FormatConverter**: Converts video format for display 
- **Holoviz**: Displays the received video stream

## Pipeline

```
VideoMasterSource → FormatConverter → Holoviz
```

This is a simplified subset of the endoscopy tool tracking application with AI processing and overlay components removed.

## Requirements

This application uses the DELTACAST.TV capture card for input stream. Contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on how to access the SDK and setup your environment.

## Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

To build with the Deltacast VideoMaster operator:
```bash
./holohub build --build-with deltacast_videomaster deltacast_receiver
```

Note that this application requires providing the VideoMaster_SDK_DIR if it is not located in a default location on the system:
```bash
./holohub build --local deltacast_receiver --configure-args="-DVideoMaster_SDK_DIR=<Path to VideoMasterSDK>"
```

## Run Instructions

From the build directory, run the command:

```bash
./applications/deltacast_receiver/cpp/deltacast_receiver
```

Or with a custom configuration:

```bash
./applications/deltacast_receiver/cpp/deltacast_receiver custom_config.yaml
```

## Configuration

The application can be configured via the YAML configuration file. Key parameters include:

### Deltacast Source Parameters
- `board`: Index of the DELTACAST.TV board to use (default: 0)
- `input`: Index of the RX channel to use (default: 0)
- `width`: Width of the input stream (default: 3840)
- `height`: Height of the input stream (default: 2160)
- `progressive`: Progressive scan mode (default: true)
- `framerate`: Frame rate of the input signal (default: 30)
- `rdma`: Enable RDMA (default: false)

### Display Parameters
- `window_title`: Title of the display window
- `display_name`: Display device name
- `framerate`: Display frame rate
