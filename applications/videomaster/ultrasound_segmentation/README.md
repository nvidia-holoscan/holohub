# Ultrasound Bone Scoliosis Segmentation

Full workflow including a generic visualization of segmentation results from a spinal scoliosis segmentation model of ultrasound videos. The model used is stateless, so this workflow could be configured to adapt to any vanilla DNN model. 

### Requirements

The provided applications are configured to use the DELTACAST.TV capture card for input stream. Contact [DELTACAST.TV](https://www.deltacast.tv/) for more details on their products and how to setup your environment.

### Data

[📦️ (NGC) Sample App Data for AI-based Bone Scoliosis Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_ultrasound_sample_data)

### Build Instructions

See instructions from the top level README.

### Run Instructions

Go to the build directory and run the command:

  ```bash
  ./applications/videomaster/ultrasound_segmentation/videomaster_ultrasound_segmentation
  ```