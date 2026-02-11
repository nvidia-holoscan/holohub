# RADAR signal processing using PVA

## Overview

The PVA Radar Operator demonstrates RADAR signal processing offload using NVIDIA's PVA (Programmable Vision Accelerator) on Holoscan. This operator is designed to run on both Jetson devices and x86 hosts, leveraging the PVA-SDK and associated solutions provided by NVIDIA to accelerate radar processing pipelines.

## Features

- **RADAR Signal Processing:** Offloads heavy-lifting signal and tensor computations to the onboard PVA hardware.
- **PVA-SDK Integration:** Integrates with NVIDIA's PVA-SDK for direct access to low-level accelerator APIs.
- **Flexible Deployment:** Supports running and prototyping on both x86 (emulator) and Jetson (Arm) devices with the same code base.
- **Portable Containerized Workflow:** Uses Docker for environment reproducibility, isolating dependencies for reliable deployment.

## Building the Container

This folder contains a `Dockerfile` to build a development container with all necessary dependencies, including the PVA-SDK and sample RADAR applications. Building the container is recommended for streamlined setup and compatibility.

To build the container, use the [HoloHub CLI](../../README.md). From the root of your HoloHub repository, run:

```sh
holohub build-container --docker-file operators/pva_radar/Dockerfile --img <my-image-name:tag>
```

This command will:

- Copy local PVA-SDK and PVA solutions debian packages from `operators/pva_radar/deps/` (see comments in `Dockerfile` for details)
- Build a container image targeting either x86 (amd64) or Jetson (arm64) depending on your build host
- Prepare all tools and environment variables for development and running PVA radar operators

> **Note:** Ensure you have placed the necessary PVA-SDK `.deb` files, pva-solutions source code tarball and pre-built `.deb` files inside `operators/pva_radar/deps/` before building the container, as described in the Dockerfile.

## Usage

Once the container is built, you can launch it for development:

```sh
holohub run-container --no-docker-build --img <my-image-name:tag>
```

Once inside the container, refer to the sample [pva_radar_pipeline](../../applications/pva_radar_pipeline/README.md) application and operator source code to develop and test radar applications using PVA.

## Folder Contents

- `Dockerfile`: Recipe for building a complete development container.
- `deps/`: Place to store required PVA-SDK and pva-solutions packages/tarballs.
- `pva_radar`: Implementation of the PVA Radar signal processing operator.
- `raw_radar_cube_source`: Operator that loads sample data from files, imitating a RADAR sensor.
- `pva_radar_graphics`: Helper operator to convert from NVCVTensorHandle outputs from the pva_radar operator to holoscan
  graphics buffers that can be rendered with holoviz operators.

## Requirements

- Access to the NVIDIA PVA-SDK and pva-solutions packages. See the [PVA documentation](https://developer.nvidia.com/embedded/pva) for details.
- A Jetson IGX or AGX, Orin or newer.
-