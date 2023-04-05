# Deltacast Endoscopy Tool Tracking

This application application is based on the endoscopy_tool_tracking application and demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream using Deltacast's videomaster SDK.

### Requirements

This application uses the DELTACAST.TV capture card for input stream. Contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on how access the SDK and to setup your environment.

### Data

This applications uses the dataset from the endoscopy tool tracking application:

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

### Build Instructions

See instructions from the top level README on how to build this application.
Note that this application requires to provide the VideoMaster_SDK_DIR if it is not located in a default location on the sytem.
This can be done with the following command, from the top level Holohub source directory:

```bash
./run build deltacast_endoscopy_tool_tracking --configure-args -DVideoMaster_SDK_DIR=<Path to VideoMasterSDK>
```

### Run Instructions

From the build directory, run the command:

```bash
./applications/deltacast_endoscopy_tool_tracking/deltacast_endoscopy_tool_tracking --data <holohub_data_dir>/endoscopy
```
