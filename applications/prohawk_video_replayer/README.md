# Prohawk video replayer

This example demonstrates how to use the Prohawk video processing operator on a pre-recorded video.

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
  
Inside the container run the application:

  `./run launch prohawk_video_replayer`