# Prohawk video replayer

This application demonstrates how to use the Prohawk video processing operator on a pre-recorded video.
This application receives video from a previously recorded file via Holoscan's video_stream_replayer operator, and then works to enhance 
the imagery so that additional details may be seen in the video output. 
This application and operator provide various preset and detailed enhancement parameters that can be changed by the command menu display.

## Data

The following dataset is used by this application:
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

For technical support or other assistance, please don't hesitate to visit us at [https://prohawk.ai/contact](https://prohawk.ai/contact)
