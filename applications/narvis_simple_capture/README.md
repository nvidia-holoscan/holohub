# Narvis Simple Capture

Visualizes frames as colored pointclouds captured from an Intel RealSense camera.
![](screenshot.png)<br>

# Build and Run
This application requires an Intel RealSense camera.

At the top level of the holohub run the following command:

```bash
./dev_container build_and_run narvis_simple_capture
```


```bash
./dev_container build_and_run narvis_simple_capture --docker_file ./applications/narvis_simple_capture/Dockerfile --base_img nvcr.io/nvidia/clara-holoscan/holoscan:v2.5.0-dgpu --img narvis_simple_capture
```