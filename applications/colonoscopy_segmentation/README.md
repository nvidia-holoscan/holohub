# Colonoscopy Polyp Segmentation

Full workflow including a generic visualization of segmentation results from a polyp segmentation models.

### Requirements

- Python 3.8+
- The provided applications are configured to either use the AJA capture card for input stream, the YUAN capture card, or a pre-recorded video of the ultrasound data (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI Colonoscopy Segmentation of Polyps](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_colonoscopy_sample_data)

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

* Using a pre-recorded video
    ```bash
    cd <HOLOHUB_SOURCE_DIR>/applications/colonoscopy_segmentation
    python3 colonoscopy_segmentation.py --source=replayer --data=<DATA_DIR>/colonoscopy_segmentation
    ```

* Using an AJA card
    ```bash
    cd <HOLOHUB_SOURCE_DIR>/applications/colonoscopy_segmentation
    python3 colonoscopy_segmentation.py --source=aja
    ```

* Using an YUAN card
    ```bash
    cd <HOLOHUB_SOURCE_DIR>/applications/colonoscopy_segmentation
    python3 colonoscopy_segmentation.py --source=qcap
    ```

### Holoscan SDK version

Colonoscopy segmentation application in HoloHub requires version 0.6+ of the Holoscan SDK.
If the Holoscan SDK version is 0.5 or lower, following code changes must be made in the application:

* In python/CMakeLists.txt: update the holoscan SDK version from `0.6` to `0.5`
* In python/multiai_ultrasound.py: `InferenceOp` is replaced with `MultiAIInferenceOp`
