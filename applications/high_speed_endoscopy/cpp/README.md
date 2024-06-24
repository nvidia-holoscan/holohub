# High-Speed Endoscopy

The application showcases how high resolution cameras can be used to capture the scene, post-processed on GPU and displayed at high frame rate.

### Requirements

This application requires:
1. an Emergent Vision Technologies camera (see [setup instructions]((https://docs.nvidia.com/holoscan/sdk-user-guide/emergent_setup.html)
2. a NVIDIA ConnectX SmartNIC with Rivermax SDK and drivers installed (see [prerequisites](../../README.md#prerequisites))
3. a display with high refresh rate to keep up with the camera's framerate
4. [additional setups](https://docs.nvidia.com/holoscan/sdk-user-guide/additional_setup.html) to reduce latency

### Build Instructions

Please refer to the top level Holohub README.md file for information on how to build this application.

> ⚠️ At this time, camera controls are hardcoded within the `gxf_emergent_source` extension. To update them at the application level, the GXF extension, and the application need to be rebuilt.
For more information on the controls, refer to the [EVT Camera Attributes Manual](https://emergentvisiontec.com/resources/?tab=umg)

### Run Instructions

First, go in your `build` or `install` directory. Then, run the commands of your choice:

* RDMA disabled
    ```bash
    # C++
    sed -i -e 's#rdma:.*#rdma: false#' ./applications/high_speed_endoscopy/cpp/high_speed_endoscopy.yaml \
        && sudo ./applications/high_speed_endoscopy/cpp/high_speed_endoscopy
    ```

* RDMA enabled
    ```bash
    # C++
    sed -i -e 's#rdma:.*#rdma: true#' ./applications/high_speed_endoscopy/cpp/high_speed_endoscopy.yaml \
        && sudo MELLANOX_RINGBUFF_FACTOR=14 ./applications/high_speed_endoscopy/cpp/high_speed_endoscopy
    ```

> ℹ️ The `MELLANOX_RINGBUFF_FACTOR` is used by the EVT driver to decide how much BAR1 size memory would be used on the dGPU. It can be changed to different number based on different use cases.
