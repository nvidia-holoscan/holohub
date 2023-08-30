# Video Replayer

Minimal example to demonstrate the use of the video stream replayer operator to load video from disk. The video frames need to have been converted to a gxf entity format, as shown [here](../../scripts/README.md#convert_video_to_gxf_entitiespy).

> Note: Support for H264 stream support is in progress

## Data

The following dataset is used by this example:
[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data).

##  Building the application

The easiest way to build this application is to use the provided Docker file.

From the Holohub main directory run the following command:

  `./dev_container build --docker_file applications/prohawk_video_replayer/Dockerfile --img holohub:prohawk`

Then launch the container to build the application:

  `./dev_container launch --img holohub:prohawk`

Inside the container build the application:

  `./run build prohawk_video_replayer`
  