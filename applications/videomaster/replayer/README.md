# Replayer

These applications demonstrate the use of custom components for replaying a video file through a dedicated IO device.

### Requirements

The provided applications are configured to use the DELTACAST.TV capture card for output stream, or a pre-recorded endoscopy video (replayer). Contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on their products.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

### Build Instructions

Built with the SDK, see instructions from the top level README.

### Run Instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the commands of your choice:

* Using an DELTACAST.TV card
    ```bash
    # C++
    ./apps/replayer/cpp/replayer


> ℹ️ The python app can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment (automatically done by `./run launch`).