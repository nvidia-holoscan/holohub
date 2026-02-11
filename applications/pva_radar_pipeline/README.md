# PVA-Accelerated Radar Pipeline

This application demonstrates the usage of the [Programmable Vision Accelerator (PVA)](#about-pva) to process
raw samples from a FMCW RADAR array like the kind used for perception in autonomous vehicles.

## About PVA

PVA is a highly power-efficient VLIW processor integrated into NVIDIA Jetson platforms, specifically designed for advanced image processing and computer vision algorithms. The pva-solutions library provides example implementations of optimized operators targeting PVA using the Compute Unified PVA (cupva) programming model provided by the PVA-SDK. For more about pva-solutions and the PVA-SDK and how to access them, see
[https://developer.nvidia.com/embedded/pva](https://developer.nvidia.com/embedded/pva).

For details about the different stages of the RADAR processing pipeline, see the [pva-solutions documentation](https://docs.nvidia.com/pva/solutions/0.4.0/pipelines/radar/radar_pipeline.html).

## Prerequisites

See the instructions in [pva_radar operators](../../operators/pva_radar/README.md) for how to set up your Holohub
environment prior to using this application.

The application is intended to be used on a Jetson device (AGX or IGX flavor, Orin or Thor generation) to experience
running the operators on actual PVA hardware. The application can also run on an x86_64 host using the PVA emulator
included with the PVA-SDK. The x86_64 environment is a useful tool for rapid development of PVA applications, debugging
program logic, and SIMD algorithm verification.  However, the x86 emulator is not very performant.

## Usage

Launch the container you built by following the pva_radar operator instructions. Then, simply run:

```sh
./holohub run pva_radar_pipeline
```

Two windows will appear. One shows the range-doppler signal after Non-coherent Integration (NCI).
The other shows a 3D point cloud of positive radar detections.

![RADAR processing using PVA](./images/pva_radar_pipeline.gif)

The pva-solutions 0.4 source package comes with one frame of sample data included. Contact NVIDIA to access more frames
of sample data and information about compatible RADAR systems.

## Troubleshooting

If you encounter an error from the holoviz operator such as "no compatible devices found", make sure your holohub
environment has basic access to the GPU, the DISPLAY environment variable is set correctly for a working display, and
you are able to run basic graphics samples like [holoviz_srgb](../holoviz/holoviz_srgb/).

If some libraries could not be found at runtime on the Jetson target device, try adding the LD_LIBRARY_PATH variable
before the run command:

```sh
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra ./holohub run pva_radar_pipeline
```

If you encounter errors on the Jetson target device related to the PVA hardware, make sure you are
using a supported Jetson platform. Jetson NX flavor and platforms older than Orin generation do not
support the PVA-SDK. You may also need to add additional device mapping parameters when launching the
container:

```sh
./holohub run-container --no-docker-build --img <my-image-name:tag> \
  --docker-opts='--device /dev/nvhost-ctrl-pva0:/dev/nvhost-ctrl-pva0 --device /dev/nvmap:/dev/nvmap --device /dev/dri/renderD128:/dev/dri/renderD128'
```

If you see an application authentication error from PVA, you can temporarily disable authentication
as a user with root privilege outside of the container (write 1 to re-enable when done testing).

```sh
sudo bash -c "echo 0 > /sys/kernel/debug/pva0/vpu_app_authentication"
```

Also see the [PVA-SDK FAQ](https://docs.nvidia.com/pva/sdk/2.8.0/frequently-asked-questions.html)
and the section on [VPU Application Signing](https://docs.nvidia.com/pva/sdk/2.8.0/vpu-allowlist.html)
for more information about how to deploy signed PVA binaries.
