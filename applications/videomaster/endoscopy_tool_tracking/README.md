# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

### Requirements

The provided applications are configured to either use the AJA or DELTACAST.TV capture card for input stream, or a pre-recorded endoscopy video (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card or contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on their products.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

### Build Instructions

Built with the SDK, see instructions from the top level README.

### Run Instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the commands of your choice:

* Using a pre-recorded video
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: replayer#' ./apps/endoscopy_tool_tracking/cpp/app_config.yaml \
      && ./apps/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking

    # Python
    python3 ./apps/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --source=replayer
    ```

* Using an AJA card
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: aja#' ./apps/endoscopy_tool_tracking/cpp/app_config.yaml \
      && ./apps/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking

    # Python
    python3 ./apps/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --source=aja
    ```

* Using a DELTACAST.TV card
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: videomaster#' ./apps/endoscopy_tool_tracking/cpp/app_config.yaml \
      && ./apps/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking

    # Python
    python3 ./apps/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --source=videomaster
    ```

> ℹ️ The python app can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment (automatically done by `./run launch`).