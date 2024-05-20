# Welcome to NVIDIA HoloHub!

![Lint](https://img.shields.io/github/actions/workflow/status/nvidia-holoscan/holohub/check_lint.yml?branch=main&label=Lint
)
![Metadata](https://img.shields.io/github/actions/workflow/status/nvidia-holoscan/holohub/check_metadata.yml?branch=main&label=Metadata
)
[![Pages](https://img.shields.io/github/actions/workflow/status/nvidia-holoscan/holohub/generate_pages.yml?branch=main&label=Pages)](https://nvidia-holoscan.github.io/holohub/)

[![Applications](https://img.shields.io/badge/Applications-64-59A700)](https://github.com/nvidia-holoscan/holohub/tree/main/applications)
[![Operators](https://img.shields.io/badge/Operators-45-59A700)](https://github.com/nvidia-holoscan/holohub/tree/main/operators)
[![Tutorials](https://img.shields.io/badge/Tutorials-7-59A700)](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials)

HoloHub is a central repository for the NVIDIA Holoscan AI sensor processing community to share apps and extensions. We invite users and developers of extensions and applications for the Holoscan Platform to reuse and contribute components and sample applications.

Visit the [HoloHub landing page](https://nvidia-holoscan.github.io/holohub/) for details on available HoloHub projects.

# Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building HoloHub](#container-build-recommended)
- [Running Applications](#running-applications)
- [Contributing to HoloHub](#contributing-to-holohub)
- [Glossary](#glossary)
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

If you prefer to build locally without `docker`, take a look at our [Native Build](./doc/developer.md#native-build) instructions.

Once your development environment is configured you may move on to [Building Applications](#building-applications).

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

See also: [Advanced Build Options](./doc/developer.md#advanced-build-options-container)

### Launch HoloHub (Container)

Launch the HoloHub Docker container environment:

```
holohub$ ./dev_container launch
```

You are now ready to run HoloHub applications! You may jump to the [Running Applications](#running-applications) section to get started.

***Note***  The `launch` option will use the default development container built using Holoscan SDK's container from NGC for the local GPU. The script will also inspect for available video devices (V4L2, AJA capture boards, Deltacast capture boards) and the presence of Deltacast's Videomaster SDK and map it into the development container.

See also: [Advanced Launch Options](#advanced-launch-options-container)

### Platform Notes (Container)

The development container has been tested on the following platforms:
- x86_x64 workstation with multiple RTX GPUs
- Clara AGX Dev Kit (dGPU mode)
- IGX Orin Dev Kit (dGPU and iGPU mode)
- AGX Orin Dev Kit (iGPU)

***Notes for AGX Orin Dev Kit***:

(1) On AGX Orin Dev Kit the launch script will add ```--privileged``` and ```--group-add video``` to the docker run command for the HoloHub sample apps to work. Please also make sure that the current user is member of the group video.

(2) When building Holoscan SDK on AGX Orin Dev Kit from source please add the option  ```--cudaarchs all``` to the ```./run build``` command to include support for AGX Orin's iGPU.

# Building Applications

_Make sure you have installed the [prerequisites](#prerequisites) before attempting to build HoloHub applications._

Sample applications based on the Holoscan platform may be found under the [applications](./applications/) directory. Sample applications are a subset of the HoloHub applications and are maintained by Holoscan SDK developers to provide a demonstration of the SDK capabilities.

HoloHub provides a convenient `run` script to build and run applications in your development environment. To build all sample applications in your development environment:

```bash
  ./run build
```

When the build is successful you can [run HoloHub applications](#running-applications).

# Running Applications

To list all available applications you can run the following command:
```bash
  ./run list
```

Then you can run the application using the command:
```bash
  ./run launch <application>
```

For example, to run the tool tracking endoscopy application:
```bash
  ./run launch endoscopy_tool_tracking
```

Several applications are implemented in both C++ and Python programming languages.
You can request a specific implementation as a trailing argument to the `./run launch` command
or omit the argument to use the default language.
For instance, the following command will run the Python implementation of the tool tracking
endoscopy application:

```bash
  ./run launch endoscopy_tool_tracking python
```

The run script reads the "run" command from the metadata.json file for a given application and runs from the "workdir" directory.
Make sure you build the application (if applicable) before running it.

# Cleanup

We recommend running the command below to reset your build directory between building Holohub applications with different configurations:

```sh
./run clear_cache
```

In some cases you may also want to clear out datasets downloaded by HoloHub apps to the `data` folder:
```sh
rm -rf ./data
```

Note that many HoloHub applications supply custom container environments with build and runtime dependencies.
Failing to clean the build cache between different applications may result in unexpected behavior where build
tools or libraries appear to be broken or missing. Clearing the build cache is a good first check to address those issues.

# Contributing to HoloHub

The goal of HoloHub is to allow engineering teams to easily contribute and share new functionalities
and to demonstrate applications. Please review the [HoloHub Contributing Guidelines](./CONTRIBUTING.md) for more information.

# Glossary

Many applications use the following keyword definitions in their README descriptions:

- `<HOLOHUB_SOURCE_DIR>` : Path to the source directory of HoloHub
- `<HOLOHUB_BUILD_DIR>` : Path to the build directory for Holohub
- `<HOLOSCAN_INSTALL_DIR>` : Path to the installation directory of Holoscan SDK
- `<DATA_DIR>` : Path to the top level directory containing the datasets for the Holohub applications
- `<MODEL_DIR>` : Path to the directory containing the inference model(s)

# Useful Links

- [Holoscan GitHub organization](https://github.com/nvidia-holoscan)
- [Holoscan SDK repository](https://github.com/nvidia-holoscan/holoscan-sdk)
