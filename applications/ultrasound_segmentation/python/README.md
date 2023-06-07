# Ultrasound Bone Scoliosis Segmentation

Full workflow including a generic visualization of segmentation results from a spinal scoliosis segmentation model of ultrasound videos. The model used is stateless, so this workflow could be configured to adapt to any vanilla DNN model. 

### Requirements

- Python 3.8+
- The provided applications are configured to either use the AJA capture card for input stream, the YUAN capture card, or a pre-recorded video of the ultrasound data (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Bone Scoliosis Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_ultrasound_sample_data)

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
    cd <HOLOHUB_SOURCE_DIR>/applications/ultrasound_segmentation/python
    python3 ultrasound_segmentation.py --source=replayer --data <DATA_DIR>/ultrasound_segmentation
    ```

* Using an AJA card
    ```bash
    cd <HOLOHUB_SOURCE_DIR>/applications/ultrasound_segmentation/python
    python3 ultrasound_segmentation.py --source=aja
    ```

* Using a YUAN card
    ```bash
    cd <HOLOHUB_SOURCE_DIR>/applications/ultrasound_segmentation/python
    python3 ultrasound_segmentation.py --source=qcap
    ```
