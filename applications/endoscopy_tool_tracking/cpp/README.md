# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

### Requirements

The provided applications are configured to either use the AJA capture card for input stream, the YUAN capture card, or a pre-recorded endoscopy video (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.


### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

### Run Instructions

In your `build` directory, run the commands of your choice:

* Using a pre-recorded video
    ```bash
    sed -i -e 's#^source:.*#source: replayer#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking --data <data_dir>/endoscopy
    ```

* Using an AJA card
    ```bash
    sed -i -e 's#^source:.*#source: aja#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking
    ```

* Using a YUAN card
    ```bash
    sed -i -e 's#^source:.*#source: qcap#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking
    ```
