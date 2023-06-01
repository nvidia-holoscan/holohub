# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

### Requirements

- Python 3.8+
- The provided applications are configured to either use the AJA capture card for input stream, the YUAN capture card, or a pre-recorded endoscopy video (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.

### Run Instructions

To run this application, you'll need to configure your PYTHONPATH environment variable to locate the
necessary python libraries based on your Holoscan SDK installation type.

If your Holoscan SDK installation type is:

* python wheels:

  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOHUB_BUILD_DIR>/python/lib
  ```

* otherwise:
 
  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
  ```
 
Next, run the commands of your choice:

This application should **be run in the build directory of Holohub** in order to load the GXF extensions.
Alternatively, the relative path of the extensions in the corresponding yaml file can be modified to match path of
the working directory.

* Using a pre-recorded video
    ```bash
    cd <HOLOHUB_BUILD_DIR>
    python3 <HOLOHUB_SOURCE_DIR>/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --source=replayer --data=<DATA_DIR>/endoscopy
    ```

* Using an AJA card
    ```bash
    cd <HOLOHUB_BUILD_DIR>
    python3  <HOLOHUB_SOURCE_DIR>/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --source=aja
    ```

* Using a YUAN card
    ```bash
    cd <HOLOHUB_BUILD_DIR>
    python3  <HOLOHUB_SOURCE_DIR>/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --source=qcap
    ```
