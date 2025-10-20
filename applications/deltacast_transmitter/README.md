# Deltacast Videomaster Transmitter

This application demonstrates the use of videomaster_transmitter to transmit a video stream through a dedicated IO device.

### Requirements

This application uses the DELTACAST.TV capture card for input stream. Contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on how access the SDK and to setup your environment.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

See instructions from the top level README on how to build this application.
Note that this application requires to provide the VideoMaster_SDK_DIR if it is not located in a default location on the system.
This can be done with the following command, from the top level Holohub source directory:

```bash
./holohub build --local deltacast_transmitter --configure-args="-DVideoMaster_SDK_DIR=<Path to VideoMasterSDK>"
```

### Run Instructions

From the build directory, run the command:

```bash
./applications/deltacast_transmitter/deltacast_transmitter --data <holohub_data_dir>/endoscopy
```
