# Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.

## Requirements

The provided applications are configured to either use capture cards for input stream, or a pre-recorded endoscopy video (replayer).

Follow the [setup instructions from the user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

To use the **Deltacast VideoMaster** capture card:

- Install a DELTACAST.TV SDI or HDMI VideoMaster card in your system.
- Obtain the VideoMaster SDK from [DELTACAST.TV](https://www.deltacast.tv/) and follow their installation instructions.
- Holoscan SDK ≥ 4.2.0 is required.
- The `holoscan-deltacast` operator module is fetched automatically when building with Deltacast support; no manual checkout is needed. See the [Deltacast VideoMaster input](../README.md#deltacast-videomaster-input) section of the application README for full details.

Refer to the Yuan documentation to use the Yuan QCap capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.

### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

To build with Deltacast VideoMaster support (requires a Deltacast card and SDK — see [Requirements](#requirements)):

```bash
./holohub build endoscopy_tool_tracking --mode deltacast
```

### Run Instructions

In your `build` directory, run the commands of your choice:

- Using a pre-recorded video

    ```bash
    sed -i -e 's#^source:.*#source: replayer#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking --data <data_dir>/endoscopy
    ```

- Using a vtk_renderer instead of holoviz

    ```bash
    sed -i -e 's#^visualizer:.*#visualizer: "vtk"#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking --data <data_dir>/endoscopy
    ```

- Using a holoviz instead of vtk_renderer

    ```bash
    sed -i -e 's#^visualizer:.*#visualizer: "holoviz"#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking --data <data_dir>/endoscopy
    ```

- Using an AJA card

    ```bash
    sed -i -e 's#^source:.*#source: aja#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking
    ```

- Using a Deltacast card

1. For Deltacast hardware: contact DELTACAST.TV for Deltacast VideoMaster SDK access. Install the SDK and set the path in your host environment.

```bash
export DELTACAST_SDK_DIR=/path/to/deltacast-sdk
```

1. Update [endoscopy_tool_tracking.yaml](./endoscopy_tool_tracking.yaml) with your target framerate
2. Build and run the containerized application from presets.

```bash
./holohub run endoscopy_tool_tracking deltacast --language=cpp
```

**For developers without Deltacast capture card hardware**: Use the mock VideoMaster SDK for CPU-based development.

```bash
./holohub run endoscopy_tool_tracking deltacast_mock --language=python
```

See the Holoscan Deltacast external module repository for more details: <https://github.com/deltacasttv/holoscan-modules>

- Using a Yuan card

    ```bash
    sed -i -e '/^#.*yuan_qcap/s/^#//' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    sed -i -e 's#^source:.*#source: yuan#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking
    ```

- Using an AJA card with hardware keying overlay (Only specific cards support this feature)

    ```bash
    ./holohub run endoscopy_tool_tracking --language=cpp --run-args=-capplications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking_aja_overlay.yaml
    ```

- Using the Slang shader operator for post-processing

    ```bash
    sed -i -e 's#^postprocessor:.*#postprocessor: slang_shader#' applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml
    applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking
    ```
