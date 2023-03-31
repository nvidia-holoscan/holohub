# Ultrasound Bone Scoliosis Segmentation

Full workflow including a generic visualization of segmentation results from a spinal scoliosis segmentation model of ultrasound videos. The model used is stateless, so this workflow could be configured to adapt to any vanilla DNN model. 

### Requirements

The provided applications are configured to either use the AJA capture card for input stream, or a pre-recorded video of the ultrasound data (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

### Data

[📦️ (NGC) Sample App Data for AI-based Bone Scoliosis Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_ultrasound_sample_data)

### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

### Run Instructions

In your build directory, run the commands of your choice:

* Using a pre-recorded video
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: replayer#' ./applications/ultrasound_segmentation/cpp/ultrasound_segmentation.yaml \
      && ./applications/ultrasound_segmentation/cpp/ultrasound_segmentation
    ```

* Using an AJA card
    ```bash
    # C++
    sed -i -e 's#^source:.*#source: aja#' ./applications/ultrasound_segmentation/cpp/ultrasound_segmentation.yaml \
      && ./applications/ultrasound_segmentation/cpp/ultrasound_segmentation
    ```
