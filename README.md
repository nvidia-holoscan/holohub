# Welcome to NVIDIA HoloHub!

![Lint](https://img.shields.io/github/actions/workflow/status/nvidia-holoscan/holohub/check_lint.yml?branch=main&label=Lint
)
![Metadata](https://img.shields.io/github/actions/workflow/status/nvidia-holoscan/holohub/check_metadata.yml?branch=main&label=Metadata
)
[![Pages](https://img.shields.io/github/actions/workflow/status/nvidia-holoscan/holohub/generate_pages.yml?branch=main&label=Pages)](https://nvidia-holoscan.github.io/holohub/)

[![Applications](https://img.shields.io/badge/Applications-61-59A700)](https://github.com/nvidia-holoscan/holohub/tree/main/applications)
[![Operators](https://img.shields.io/badge/Operators-38-59A700)](https://github.com/nvidia-holoscan/holohub/tree/main/operators)

HoloHub is a central repository for the NVIDIA Holoscan AI sensor processing community to share apps and extensions. We invite users and developers of extensions and applications for the Holoscan Platform to reuse and contribute components and sample applications.

Visit the [HoloHub landing page](https://nvidia-holoscan.github.io/holohub/) for details on available HoloHub projects.

# Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building HoloHub](#building-holohub)
- [Running applications](#running-applications)
- [Contributing to HoloHub](#contributing-to-holohub)
- [Useful Links](#useful-links)

# Overview

HoloHub is a collection of applications and extensions created by the Holoscan AI sensor processing community.
The following directories make up the core of HoloHub:

- Example applications: Visit [`applications`](./applications) to explore an evolving collection of example
  applications built on the NVIDIA Holoscan platform. Examples are available from NVIDIA, partners, and
  community collaborators.
- Community components: Visit [`operators`](./operators/) and [`gxf_extensions`](./gxf_extensions) to explore
  reusable Holoscan modules.
- Tutorials: Visit [`tutorials`](./tutorials/) for extended walkthroughs and tips for the Holoscan platform.

Visit the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/overview.html) to learn more about the NVIDIA Holoscan AI sensor processing platform.

# Prerequisites

## Supported Platforms

You will need a platform supported by NVIDIA Holoscan SDK. Refer to the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#prerequisites) for the latest requirements. In general, Holoscan supported platforms include:
- An x64 PC with an Ubuntu operating system and an NVIDIA GPU; or
- A supported NVIDIA ARM development kit.

Individual examples and operators in HoloHub may have additional platform requirements. For instance, some examples may support only ARM platforms.

## Build Environment

You may choose to build HoloHub in a containerized development environment or in your native environment.

We strongly recommend new users follow our [Container Build](#container-build-recommended) instructions to set up a container for development.

If you prefer to build locally without `docker`, jump to [Native Build](#native-build-optional) instructions.

Once your development environment is configured you may move on to [Building Sample Applications](#building-sample-applications).

## Container Build (Recommended)

### Software Prerequisites (Container)

To build and run HoloHub in a containerized environment you will need:
  - the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (v1.12.2 or later)
  - [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository), including the buildx plugin (`docker-buildx-plugin`)
  - `git` version control

You will also need to set up your NVIDIA NGC credentials at [ngc.nvidia.com](https://catalog.ngc.nvidia.com/).

### Fetch HoloHub (Container)
  
Clone the HoloHub repository to your local system:
```sh
$ git clone https://www.github.com/nvidia-holoscan/holohub.git
```

Alternatively, download HoloHub sources as a ZIP archive from the HoloHub GitHub homepage.

### Build HoloHub (Container)

Simply run the following commands to build the development container. The build may take a few minutes.

```sh
$ cd holohub
holohub$ ./dev_container build
```

Check to verify that the image is created:
```bash
user@ubuntu-20-04:/media/data/github/holohub$ docker images
REPOSITORY                               TAG           IMAGE ID       CREATED         SIZE
holohub                                  ngc-v0.6.0-dgpu   b6d86bccdcac   9 seconds ago   8.47GB
nvcr.io/nvidia/clara-holoscan/holoscan   v0.6.0-dgpu       1b4df7733d5b   5 weeks ago     8.04GB
```

***Note:*** The development container script ```dev_container``` will by default detect if the system is using an iGPU (integrated GPU) or a dGPU (discrete GPU) and use [NGC's Holoscan SDK container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan) **`v1.0`** for the [Container build](#container-build-recommended). See [Advanced Container Build Options](#advanced-build-options-container) if you would like to use an older version of the SDK as a custom base image.

See also: [Advanced Build Options](#advanced-build-options-container)

### Launch HoloHub (Container)

Launch the HoloHub Docker container environment:

```
holohub$ ./dev_container launch
```

You are now ready to run HoloHub applications! You may jump to the [Running Applications](#running-applications) section to get started.

***Note***  The `launch` option will use the default development container built using Holoscan SDK's container from NGC for the local GPU. The script will also inspect for available video devices (V4L2, AJA capture boards, Deltacast capture boards) and the presence of Deltacast's Videomaster SDK and map it into the development container.

See also: [Advanced Launch Options](#advanced-launch-options-container)

### Advanced Build Options (Container)

#### View All Options

Run the following to view all build options available for the HoloHub container script:
```sh
$ ./dev_container help build
```

#### Custom Base Image

You may configure a custom base image for building the HoloHub container. For instance, if you want to use a local Holoscan container as the base image, use the following command:

```bash
  ./dev_container build --base_img holoscan-sdk-build-x86_64:latest --img holohub:sdk-dev-latest
```

The command above uses the following arguments:
- `--base_img`  is used to configure the base container image;
- `--img` defines the fully qualified name of the image output by `./dev_container`.

After ```./dev_container build``` has completed ```docker images``` will list the new image:

```bash
user@ubuntu-20-04:/media/data/github/holohub$ docker images
REPOSITORY                               TAG               IMAGE ID       CREATED          SIZE
holohub                                  sdk-dev-latest    cb0231f77856   54 seconds ago   8.22GB
```

Base containers created during the Holoscan SDK build process use the following naming convention by default:

| Base Image Name | Target Architecture | Target IGX Configuration |
| --- | --- | --- |
| `holoscan-sdk-build-x86_64` | `x86_64` | N/A |
| `holoscan-sdk-build-aarch64-dgpu` | `aarch64` | dGPU mode |
| `holoscan-sdk-build-aarch64-igpu` | `aarch64` | iGPU mode |

#### Using a Custom Dockerfile

Several HoloHub applications use a custom Dockerfile to alter or extend the default HoloHub container. Use the following command to build from a custom Dockerfile:

```bash
$ ./dev_container build  --docker_file <path_to_dockerfile>  --img holohub-debug:local-sdk-v0.6.0
```

Where:
- `--docker_file`  is the path to the container's Dockerfile;
- `--img` defines the fully qualified image name.

#### Build with Verbose Output

To print the values for base image, Dockerfile, GPU type, and output image name, use ```--verbose```.

For example, on an x86_64 system with dGPU, the default build command will print the following values when using the ```--verbose``` option.

```bash
user@ubuntu-20-04:/media/data/github/holohub$ ./dev_container build --verbose
Build (HOLOHUB_ROOT:/media/data/github/holohub)...
Build (gpu_type_type:dgpu)...
Build (base_img:nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu)...
Build (docker_file_path:/media/data/github/holohub/Dockerfile)...
Build (img:holohub:ngc-v0.6.0-dgpu)...
....
```

### Advanced Launch Options (Container)

#### View All Options

Run the command below to view all available launch options in the `dev_container` script:

```sh
$ ./dev_container help launch
```

#### Launch a Local Holoscan SDK Container

To use a HoloHub container image built with a local Holoscan SDK container:

```bash
$ ./dev_container launch --img holohub:local-sdk-latest --local_sdk_root <path_to_holoscan_sdk>
```

#### Launch a Named HoloHub Container

To launch custom HoloHub container with fully qualified name, e.g. "holohub:ngc-sdk-sample-app"

```bash
$ ./dev_container launch --img holohub:ngc-sdk-sample-app
```

#### Forward X11 Graphics Over SSH

```bash
  ./dev_container launch --ssh_x11
```

#### Support Nsight Systems profiling in the HoloHub Container

```bash
  ./dev_container launch --nsys_profile
```

#### Print Verbose Output

```sh
  ./dev_container launch --verbose
```

For example, on an x86_64 system with dGPU ```./dev_container launch --verbose``` will print the following values.

```bash
user@ubuntu-20-04:/media/data/github/holohub$ ./dev_container launch  --verbose
2023-07-10 18:36:53 $ xhost +local:docker
non-network local connections being added to access control list
Launch (HOLOHUB_ROOT: /media/data/github/holohub)...
Launch (mount_device_opt:  --device /dev/video0:/dev/video0 --device /dev/video1:/dev/video1)...
Launch (conditional_opt:  -v /usr/lib/libvideomasterhd.so:/usr/lib/libvideomasterhd.so -v /opt/deltacast/videomaster/Include:/usr/local/deltacast/Include)...
Launch (local_sdk_opt: )...
Launch (nvidia_icd_json: /usr/share/vulkan/icd.d/nvidia_icd.json)...
Launch (image: holohub:ngc-v0.6.0-dgpu)...
....
```

Please note that the values of some of the variables will vary depending on configured options, iGPU or dGPU, availability of devices for video capture, or other environment factors.

### Platform Notes (Container)

The development container has been tested on the following platforms:
- x86_x64 workstation with multiple RTX GPUs
- Clara AGX Dev Kit (dGPU mode)
- IGX Orin Dev Kit (dGPU and iGPU mode)
- AGX Orin Dev Kit (iGPU)


***Notes for AGX Orin Dev Kit***:

(1) On AGX Orin Dev Kit the launch script will add ```--privileged``` and ```--group-add video``` to the docker run command for the HoloHub sample apps to work. Please also make sure that the current user is member of the group video.

(2) When building Holoscan SDK on AGX Orin Dev Kit from source please add the option  ```--cudaarchs all``` to the ```./run build``` command to include support for AGX Orin's iGPU.


## Native Build (Optional)

### Software Prerequisites (Native)

Refer to the [Holoscan SDK README](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/README.md) for ways to install Holoscan SDK in local environment: Debian package, Python wheels or from source.

Install the package dependencies for HoloHub on your host system. The easiest way to make sure the minimal package dependencies is to use the run script from the top level directory.

```bash
  # if sudo is available
  sudo ./run setup
```

If you prefer you can also install the dependencies manually:
- [CMake](https://www.cmake.org): 3.24.0+
- Python interpreter: 3.8 to 3.11
- Python dev: 3.8 to 3.11 (matching version of the interpreter)
- ffmpeg runtime
- [ngc-cli](https://ngc.nvidia.com/setup/installers/cli)
- wget
- CUDA: 11.6 or 11.8 (CUDA 12 is not supported yet)
- libcudnn8
- libcudnn8-dev
- libnvinfer-dev
- libnvinfer-plugin-dev
- libnvonnxparsers-dev

*Note: the run script setup installs the minimal set of dependencies required to run the sample applications. Other applications might require more dependencies. Please refer to the README of each application for more information.*

# Building Sample Applications

_Make sure you have installed the [prerequisites](#prerequisites) before attempting to build HoloHub applications._

Sample applications based on the Holoscan platform may be found under the [applications](./applications/) directory. Sample applications are a subset of the HoloHub applications and are maintained by Holoscan SDK developers to provide a demonstration of the SDK capabilities.

HoloHub provides a convenient `run` script to build and run applications in your development environment. To build all sample applications in your development environment:

```bash
  ./run build
```

When the build is successful you can [run the sample applications](#running-applications).

### Advanced Build Options

#### View Build Options

```sh
./run -h
```

#### Local SDK Path

If you have an installation of the Holoscan SDK which is not in a standard path, you may want to provide the root directory of your Holoscan SDK installation.

```bash
  ./run build --sdk <path to the Holoscan SDK installation directory>
```

#### Building a Specific Application

By default HoloHub builds all the sample applications that are maintained with the SDK. You can build specific applications by the name of the directory.

```bash
  ./run build <application>
```

For example:

```bash
  ./run build endoscopy_tool_tracking
```

Note that CMake will build the application in the directory specified. If there are multiple languages, the script will attempt to build all of them.


#### Building application or operator manually

If you prefer to build applications and operator manually you can follow the steps below.

```bash
# Export cuda (in case it's not already in the path)
export PATH=$PATH:/usr/local/cuda/bin

# Configure HoloHub with CMake
cmake -S <path_to_holohub_source>            # Source directory
      -B build                               # Build directory
      -DPython3_EXECUTABLE=/usr/bin/python3  # Specifies the python executable for CMake to find the correct version
      -DHOLOHUB_DATA_DIR=$(pwd)/data         # Specifies the data directory
      -DBUILD_SAMPLE_APPS=1                  # If you want to build the sample applications
      or
      -DAPP_<name_of_the_application>=1      # To build a specific application


# Build the application(s)
cmake --build build
```

### Additional Build Notes

While not all applications requires building HoloHub, the current build system automatically manages dependencies (applications/operators) and also downloads and converts datasets at build time.

You can refer to the README of each application/operator if you prefer to build/run them manually.

The run script creates a `data` subdirectory to store the downloaded HoloHub data.
This directory is noted `HOLOHUB_DATA_DIR/holohub_data_dir` in the documentation, READMEs and metadata files.

# Running Applications

To list all available applications you can run the following command:
```bash
  ./run list
```

Then you can run the application using
```bash
  ./run launch <application> <language>
```

For example, to run the tool tracking endoscopy application in C++
```bash
  ./run launch endoscopy_tool_tracking cpp
```

and to run the same application in python:
```bash
  ./run launch endoscopy_tool_tracking python
```

The run script reads the "run" command from the metadata.json file for a given application and runs from the "workdir" directory.
Make sure you build the application (if applicable) before running it.

### Advanced Run Options

#### Pass additional arguments to the application command

```bash
  ./run launch endoscopy_tool_tracking python --extra_args '-r visualizer'
```

#### Profile using Nsight Systems

```bash
  ./run launch endoscopy_tool_tracking python --nsys_profile
```

This will create a Nsight Systems report file in the application working directory. Information on the generated report file is printed on the end of the application log:

```
Generating '/tmp/nsys-report-bcd8.qdstrm'
[1/1] [========================100%] report8.nsys-rep
Generated:
    /workspace/holohub/build/report8.nsys-rep
```

This file can be loaded and visualized with the Nsight Systems UI application:

```bash
  nsys-ui /workspace/holohub/build/report8.nsys-rep
```

# Contributing to HoloHub

The goal of HoloHub is to allow engineering teams to easily contribute and share new functionalities
and to demonstrate applications. Please review the [HoloHub Contributing Guidelines](https://github.com/nvidia-holoscan/holohub/blob/main/CONTRIBUTING.md) for more information.

# Useful Links

- [Holoscan GitHub organization](https://github.com/nvidia-holoscan)
- [Holoscan SDK repository](https://github.com/nvidia-holoscan/holoscan-sdk)
