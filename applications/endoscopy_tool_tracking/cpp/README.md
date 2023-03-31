# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

### Requirements

The provided applications are configured to either use the AJA capture card for input stream, or a pre-recorded endoscopy video (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

Unzip the sample data:

```
unzip holoscan_endoscopy_sample_data_20230128.zip -d <data_dir>
```

### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

### Run Instructions

In your build directory, run the commands of your choice:

* Using a pre-recorded video
    ```bash
    # C++
    sed -i -e 's#applications/endoscopy_tool_tracking/data#<data_dir>#' <build_dir>/endoscopy_tool_tracking.yaml
    sed -i -e 's#^source:.*#source: replayer#' <build_dir>/endoscopy_tool_tracking.yaml
    <build_dir>/endoscopy_tool_tracking
    ```

* Using an AJA card
    ```bash
    # C++
    sed -i -e 's#applications/endoscopy_tool_tracking/data#<data_dir>#' <build_dir>/endoscopy_tool_tracking.yaml
    sed -i -e 's#^source:.*#source: aja#' <build_dir>/endoscopy_tool_tracking.yaml \
      && <build_dir>/endoscopy_tool_tracking
    ```
