# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

### Requirements

- Python 3.8+
- The provided applications are configured to either use the AJA capture card for input stream, or a pre-recorded endoscopy video (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

Unzip the sample data:

```
unzip holoscan_endoscopy_sample_data_20230128.zip -d <data_dir>
```

### Run Instructions

* (Optional) Create and use a virtual environment:

  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* Install Holoscan PyPI package

  ```
  pip install holoscan
  ```

Run the commands of your choice:

This application should **be run in the build directory of Holohub** in order to load the GXF extensions.
Alternatively, the relative path of the extensions in the corresponding yaml file can be modified to match path of
the working directory.

* Using a pre-recorded video
    ```bash
    # Python
    cd <HOLOHUB_BUILD_DIR>
    export HOLOSCAN_DATA_PATH=<DATA_DIRECTORY>
    export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
    python3 endoscopy_tool_tracking.py --source=replayer
    ```

* Using an AJA card
    ```bash
    # Python
    cd <HOLOHUB_BUILD_DIR>
    export HOLOSCAN_DATA_PATH=<DATA_DIRECTORY>
    export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
    python3 endoscopy_tool_tracking.py --source=aja
    ```

