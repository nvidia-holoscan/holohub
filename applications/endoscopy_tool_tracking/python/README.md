# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

## Requirements

- Python 3.8+
- The provided applications are configured to either use the AJA, DELTACAST or Yuan capture cards for input stream, or a pre-recorded endoscopy video (replayer).
Follow the [AJA Video Systems setup guide](../../../../operators/aja_source/setup.md) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.

### Run Instructions

To run this application, you'll need to configure your PYTHONPATH environment variable to locate the
necessary python libraries based on your Holoscan SDK installation type.

You should refer to the [glossary](../../README.md#Glossary) for the terms defining specific locations within HoloHub.

If your Holoscan SDK installation type is:

- python wheels:

  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOHUB_BUILD_DIR>/python/lib
  ```

- otherwise:

  ```bash
  export PYTHONPATH=$PYTHONPATH:<HOLOSCAN_INSTALL_DIR>/python/lib:<HOLOHUB_BUILD_DIR>/python/lib
  ```

Next, run the commands of your choice:

This application should **be run in the build directory of Holohub** in order to load the GXF extensions.
Alternatively, the relative path of the extensions in the corresponding yaml file can be modified to match path of
the working directory.

- Using a pre-recorded video

    ```bash
    cd <HOLOHUB_BUILD_DIR>
    python3 <HOLOHUB_SOURCE_DIR>/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --source=replayer --data=<DATA_DIR>/endoscopy
    ```

- Using an AJA card

    ```bash
    ./holohub run endoscopy_tool_tracking python --run-args=-s=aja
    ```

- Using a Deltacast card

1. For Deltacast hardware: contact DELTACAST.TV for Deltacast VideoMaster SDK access. Install the SDK and set the path in your host environment.

```bash
export DELTACAST_SDK_DIR=/path/to/deltacast-sdk
```

1. Update [endoscopy_tool_tracking.yaml](./endoscopy_tool_tracking.yaml) with your target framerate
2. Build and run the containerized application from presets.

```bash
./holohub run endoscopy_tool_tracking deltacast --language=python
```

**For developers without Deltacast capture card hardware**: Use the mock VideoMaster SDK for CPU-based development.

```bash
./holohub run endoscopy_tool_tracking deltacast_mock --language=python
```

See the Holoscan Deltacast external module repository for more details: <https://github.com/deltacasttv/holoscan-modules>

- Using a YUAN card

    ```bash
    cd <HOLOHUB_BUILD_DIR>
    python3  <HOLOHUB_SOURCE_DIR>/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.py --source=yuan
    ```

- Using an AJA card with hardware keying overlay (Only specific cards support this feature)

    ```bash
    ./holohub run endoscopy_tool_tracking --language python --run-args="-c=applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking_aja_overlay.yaml -s=aja"
    ```

- Using the Slang shader operator for post-processing

    ```bash
    ./holohub run endoscopy_tool_tracking --language python --run-args="-p=slang_shader"
    ```
