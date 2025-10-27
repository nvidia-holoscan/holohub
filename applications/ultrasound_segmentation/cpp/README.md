# Ultrasound Bone Scoliosis Segmentation

Full workflow including a generic visualization of segmentation results from a spinal scoliosis segmentation model of ultrasound videos. The model used is stateless, so this workflow could be configured to adapt to any vanilla DNN model. 

## Requirements

The provided applications are configured to either use the AJA capture card for input stream, or a pre-recorded video of the ultrasound data (replayer). Follow the [setup instructions from the user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/aja_setup.html) to use the AJA capture card.

## Data

[📦️ (NGC) Sample App Data for AI-based Bone Scoliosis Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_ultrasound_sample_data)

The data is automatically downloaded and converted to the correct format when building the application.
If you want to manually convert the video data, please refer to the instructions for using the [convert_video_to_gxf_entities](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) script.

## Build and Run Instructions

Please refer to the top level Holohub README.md for more information about the HoloHub CLI.

### Pre-recorded Video Input

```bash
./holohub run ultrasound_segmentation --language=cpp [--local]
```

### AJA Capture Card Input

```bash
./holohub run ultrasound_segmentation --language=cpp [--local] \
    --configure-args="-DOP_aja_source:BOOL=ON" \
    --run-args="--source=aja"
```
