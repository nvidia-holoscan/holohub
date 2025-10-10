# Deltacast Videomaster Receiver

This application demonstrates the use of videomaster_source to receive and display video streams from a Deltacast capture card using Holoviz for visualization.

## Requirements

This application uses the DELTACAST.TV capture card for input stream. Contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on how to access the SDK and setup your environment.

## Build Instructions

See instructions from the top level README on how to build this application.
Note that this application requires to provide the VideoMaster_SDK_DIR if it is not located in a default location on the system.
This can be done with the following command, from the top level Holohub source directory:

```bash
./holohub build --local deltacast_receiver --configure-args="-DVideoMaster_SDK_DIR=<Path to VideoMasterSDK>"
```

## Run Instructions

From the build directory, run the command:

```bash
./applications/deltacast_receiver/cpp/deltacast_receiver
```
