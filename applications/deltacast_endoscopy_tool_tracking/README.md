# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

### Requirements

The provided applications are configured to use the DELTACAST.TV capture card for input stream. Contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on their products and how to setup your environment.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

### Build Instructions

See instructions from the top level README.

### Run Instructions

Go to the build directory and run the command:

  ```bash
  ./applications/videomaster/endoscopy_tool_tracking/videomaster_tool_tracking
  ```