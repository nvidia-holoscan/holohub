# High Speed Endoscopy

The high speed endoscopy application showcases how high resolution cameras can be used to capture the scene, post-processed on GPU, and displayed at high frame rate.

This application requires:
1. an Emergent Vision Technologies camera (see {ref}`setup instructions<emergent-vision-tech>`)
2. an NVIDIA ConnectX SmartNIC with Rivermax SDK and drivers installed
3. a display with high refresh rate to keep up with the camera's framerate
4. {ref}`additional setups<additional_setup>` to reduce latency

> **Tip**
Tested on the Holoscan DevKits (ConnectX included) with:
- EVT HB-9000-G-C: 25GigE camera with Gpixel GMAX2509
- SFP28 cable and QSFP28 to SFP28 adaptor
- [Asus ROG Swift PG279QM](https://rog.asus.com/us/monitors/27-to-31-5-inches/rog-swift-pg279qm-model/) and [Asus ROG Swift 360 Hz PG259QNR](https://rog.asus.com/us/monitors/23-to-24-5-inches/rog-swift-360hz-pg259qnr-model/) monitors with NVIDIA G-SYNC technology



![](docs/workflow_high_speed_endoscopy_app.png)<br>
Fig. 1 Hi-Speed Endoscopy App


The data acquisition happens using `emergent-source`, by default it is set to 4200x2160 at 240Hz.
The acquired data is then demosaiced in GPU using CUDA via `bayer-demosaic` and displayed through
`holoviz-viewer`.

The peak performance that can be obtained by running these applications with the
recommended hardware, GSYNC and RDMA enabled on exclusive display mode is 10ms on
Clara AGX Devkit and 8ms on NVIDIA IGX Orin DevKit ES. This is the photon-to-glass latency of
a frame from scene acquisition to display on monitor.

## Building and Running the Application

This application cannot be compiled within the Holohub container. It needs to be compiled on the
host machine.

### Set up the environment

```bash
export LD_LIBRARY_PATH=/opt/EVT/eSDK:$LD_LIBRARY_PATH
```

### Compile the application

#### Using HoloHub CLI
HoloHub CLI command can also be used to build the application.

```bash
./holohub build high_speed_endoscopy --local
```

If the above command does not work, please file an issue and try the steps below.

#### Explicit commands on the host machine

```bash
cd <HOLOHUB_SOURCE_DIR>
rm -rf build/high_speed_endoscopy
mkdir -p build/high_speed_endoscopy
cd build/high_speed_endoscopy
cmake ../.. -DAPP_high_speed_endoscopy=ON
make -j
```

### Run the application

```bash
cd <HOLOHUB_SOURCE_DIR>
sudo ./holohub run high_speed_endoscopy --local --language cpp
```

For python version:

```bash
cd <HOLOHUB_SOURCE_DIR>
sudo ./holohub run high_speed_endoscopy --local --language python
```

## Troubleshooting

1. **Problem:** The application fails to find the EVT camera.
    - Make sure that the MLNX ConnectX SmartNIC is configured with the correct IP address. Follow section [Post EVT Software Installation Steps](https://docs.nvidia.com/holoscan/sdk-user-guide/emergent_setup.html#post-evt-software-installation-steps)

2. **Problem:** The application fails to open the EVT camera.
    - Make sure that the application was run with `sudo` privileges.
    - Make sure a valid Rivermax license file is located at `/opt/mellanox/rivermax/rivermax.lic`.
    - Make sure that a previously run `high_speed_endoscopy` application has been terminated successfully. One might need to run `sudo kill -9 <PID>` to terminate the application.

3. **Problem:** The application fails to connect to the EVT camera with error message “GVCP ack error”.
    - It could be an issue with the HR12 power connection to the camera. Disconnect the HR12 power connector from the camera and try reconnecting it.

4. **Problem:** The applications fails with a segmentation fault at runtime, in the HoloViz visualization operator.
    - It could be an issue with the configured BAR size on the GPU. Use the [display mode selector tool](https://developer.nvidia.com/displaymodeselector) to increase the BAR size.
    - If you are using a lower resolution monitor, then you might need to decrease the resolution of the vistualization in the `high_speed_endoscopy.yaml` file.


## Known Issues

- When the application is closed from the 'X' button in the visualization window, the application might not terminate gracefully and throw an error.
In this case, one might need to power-cycle the camera before running the application again.
