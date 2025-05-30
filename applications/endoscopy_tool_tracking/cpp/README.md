# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

### Requirements

The provided applications are configured to either use capture cards for input stream, or a pre-recorded endoscopy video (replayer).

Follow the [setup instructions from the user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

Refer to the Deltacast documentation to use the Deltacast VideoMaster capture card.

Refer to the Yuan documentation to use the Yuan QCap capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.


### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.
In order to build with the Deltacast VideoMaster operator use ```./run build --with deltacast_videomaster```

### Run Instructions

In your `build` directory, run the commands of your choice:

* Using a pre-recorded video
    ```bash
    sed -i -e 's#^source:.*#source: replayer#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking --data <data_dir>/endoscopy
    ```

* Using a vtk_renderer instead of holoviz
    ```bash
    sed -i -e 's#^visualizer:.*#visualizer: "vtk"#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking --data <data_dir>/endoscopy
    ```

* Using a holoviz instead of vtk_renderer
    ```bash
    sed -i -e 's#^visualizer:.*#visualizer: "holoviz"#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking --data <data_dir>/endoscopy
    ```

* Using an AJA card
    ```bash
    ./run launch endoscopy_tool_tracking cpp --extra_args -capplications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking_aja_overlay.yaml
    ```

* Using a Deltacast card
    ```bash
    sed -i -e '/^#.*deltacast_videomaster/s/^#//' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    sed -i -e 's#^source:.*#source: deltacast#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking
    ```
* Using a Yuan card
    ```bash
    sed -i -e '/^#.*yuan_qcap/s/^#//' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    sed -i -e 's#^source:.*#source: yuan#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking
    ```
