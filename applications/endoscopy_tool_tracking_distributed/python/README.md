# Distributed Endoscopy Tool Tracking

This application is similar to the Endoscopy Tool Tracking application, but the distributed version divides the application into three fragments:

1. Video Input: get video input from a pre-recorded video file.
2. Inference: run the inference using LSTM and run the post-processing script.
3. Visualization: display input video and inference results.

Based on an LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

### Requirements

- Python 3.8+
- The provided applications are configured to use a pre-recorded endoscopy video (replayer). 

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.

### Run Instructions

### Run Instructions

Run the following command to start the application.  This will run the app with a pre-recorded video as input:

```sh
./dev_container build_and_run endoscopy_tool_tracking_distributed --language python
```
