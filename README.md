# Holoscan Reference Applications

![Lint](https://img.shields.io/github/actions/workflow/status/nvidia-holoscan/holohub/check_lint.yml?branch=main&label=Lint
)
![Metadata](https://img.shields.io/github/actions/workflow/status/nvidia-holoscan/holohub/check_metadata.yml?branch=main&label=Metadata
)

Visit [https://nvidia-holoscan.github.io/holohub](https://nvidia-holoscan.github.io/holohub) for a searchable catalog of all available components.

This is a central repository for the NVIDIA Holoscan AI sensor processing community to share reference applications, operators, tutorials and benchmarks. We invite users and developers of the Holoscan platform to reuse and contribute to this repository.

# Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Building](#container-build-recommended)
- [Running](#running-applications)
- [Contributing](#contributing)
- [Glossary](#glossary)
- [More Information](#more-information)

# Overview

This repository is a collection of applications and extensions created by the Holoscan AI sensor processing community.
The following directories make up the core of this repo:

- **Example applications**: Visit [`applications`](./applications) to explore an evolving collection of example
  applications built on the NVIDIA Holoscan platform. Examples are available from NVIDIA, partners, and
  community collaborators, and provide a demonstration of the SDK capabilities.
- **Community components**: Visit [`operators`](./operators/) and [`gxf_extensions`](./gxf_extensions) to explore
  reusable Holoscan modules.
- **Package configurations**: Visit [`pkg`](./pkg/) for a list of debian package to generate, to distribute operators and applications for easier development.
- **Tutorials**: Visit [`tutorials`](./tutorials/) for extended walkthroughs and tips for the Holoscan platform.
- **Benchmarks**: Visit [`benchmarks`](./benchmarks/) for performance benchmarks, tools, and examples to evaluate the performance of Holoscan applications.


Visit the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/overview.html) to learn more about the NVIDIA Holoscan AI sensor processing platform. You can also chat with the [Holoscan-GPT](https://chatgpt.com/g/g-M6hMJimGa-holochatgpt) Large Language Model to learn about using Holoscan SDK, ask questions, and get code help. Holoscan-GPT requires an OpenAI account.

# Prerequisites

## Supported Platforms

You will need a platform supported by NVIDIA Holoscan SDK. Refer to the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#prerequisites) for the latest requirements. In general, Holoscan supported platforms include:
- An x64 PC with an Ubuntu operating system and an NVIDIA GPU; or
- A supported NVIDIA ARM development kit.

Individual examples and operators in this repo may have additional platform requirements. For instance, some examples may support only ARM platforms.

## Build Environment

You may choose to build applications and operators in a containerized development environment or in your native environment.

We strongly recommend new users follow our [Container Build](#container-build-recommended) instructions to set up a container for development.

If you prefer to build locally without `docker`, take a look at our [Native Build](./doc/developer.md#native-build) instructions.

Once your development environment is configured you may move on to [Building the Holohub components you are interested in](#building-operators-applications-and-packages).

> **NOTE:** Several applications and operators require additional dependencies beyond the basic prerequisites listed above. Please refer to the README of the specific application or operator for detailed dependency information before attempting to build or run it.

## Container Build (Recommended)

### Software Prerequisites

To build and run in a containerized environment you will need:
  - the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (v1.12.2 or later)
  - [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository), including the buildx plugin (`docker-buildx-plugin`)
  - `git` version control
  - (optional) A [Python3](https://www.python.org/downloads/) base installation to run the `holohub` infrastructure script

You will also need to set up your NVIDIA NGC credentials at [ngc.nvidia.com](https://catalog.ngc.nvidia.com/).

### Fetch

Clone the repository to your local system:
```sh
$ git clone https://www.github.com/nvidia-holoscan/holohub.git
$ cd holohub
```

Alternatively, download sources as a ZIP archive from the GitHub homepage.

### Build and run command (recommended)

The easiest way to build and run Holohub applications is to use the `./dev_container build_and_run` command.

```sh
$ ./dev_container build_and_run <application_name>
```

If you want to use a specific based image for the application, you can use the `--base_img` option.

```sh
$ ./dev_container build_and_run --base_img <base_image> <application_name>
```

> **NOTE:** The build_and_run command is not supported for all applications and operators, especially applications that requires manual configurations or applications that requires additional datasets. Please refer to the README of each application or operator for more information.

If you want a more detailed command to build and run a specific application, please follow the instructions below.

### Build

Holohub provides a default development container that can be used to build and run applications. However several applications and operator requires specific dependencies that are not available in the default development container and are provided by specific docker files. Please refer to the README of each application or operator for more information.

Run the following command to build the default development container. The build may take a few minutes.

```sh
$ ./dev_container build
```

Depending on the application or operator you are building, you may need to point to the specific docker file provided by the application or operator.

```sh
$ ./dev_container build --docker_file <path_to_the_application_dockerfile>
```

Check to verify that the image is created:
```bash
$ docker images
REPOSITORY      TAG               IMAGE ID       CREATED         SIZE
...
holohub         ngc-v3.2.0-dgpu   17e3aa51f129   13 days ago     13.2GB
...
```

***Note:*** The development container script ```dev_container``` will by default detect if the system is using an iGPU (integrated GPU) or a dGPU (discrete GPU) and use [NGC's Holoscan SDK container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan) **`v2.9`** for the [Container build](#container-build-recommended). See [Advanced Container Build Options](/doc/developer.md#advanced-build-options-container) if you would like to use an older version of the SDK as a custom base image.

See the [Developer Reference document](/doc/developer.md) for additional options.

### Launch

Launch the Docker container environment:

```
$ ./dev_container launch
```

You are now ready to [build Holohub operators, applications, or packages!](#building-operators-applications-and-packages)

***Note***  The `launch` option will use the default development container built using Holoscan SDK's container from NGC for the local GPU. The script will also inspect for available video devices (V4L2, AJA capture boards, Deltacast capture boards) and the presence of Deltacast's Videomaster SDK and map it into the development container.

See also: [Advanced Launch Options](/doc/developer.md#advanced-launch-options-container)

### Platform Notes (Container)

The development container has been tested on the following platforms:
- x86_64 workstation with multiple RTX GPUs
- Clara AGX Dev Kit (dGPU mode)
- IGX Orin Dev Kit (dGPU and iGPU mode)
- AGX Orin Dev Kit (iGPU)

***Notes for AGX Orin Dev Kit***:

(1) On AGX Orin Dev Kit the launch script will add ```--privileged``` and ```--group-add video``` to the docker run command for the reference applications to work. Please also make sure that the current user is member of the group video.

(2) When building Holoscan SDK on AGX Orin Dev Kit from source please add the option  ```--cudaarchs all``` to the ```./run build``` command to include support for AGX Orin's iGPU.

# Building Operators, Applications, and Packages

> _Make sure you have either launched your [development container](#container-build-recommended) or [set up your local environment](./doc/developer.md#native-build) before attempting to build Holohub components._

This repository provides a convenience `run` script to abstract some of the CMake build process below.

Run the following to list existing components available to build:

  ```bash
  ./run list
  ```

Then run the following to build the component of your choice, using either its name or its path:

  ```bash
  # Build using the component name
  ./run build <package|application|operator>
  # Ex: ./run build endoscopy_tool_tracking

  # Build using the component path
  ./run build ./<pkg|applications|operator>/<name>
  # Ex: ./run build ./applications/endoscopy_tool_tracking/
  ```

The build artifacts will be created under `./build/<component_name>` by default to isolate them from other components which might have different build environment requirements. You can override this behavior and other defaults, see `./run build --help` for more details.

# Running Applications

To list all available applications you can run the following command:

  ```bash
  ./run list_apps
  ```

Then you can run the application using the command:

  ```bash
  ./run launch <application>
  # Ex: ./run launch endoscopy_tool_tracking
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

You can run the command below to reset your `build` directory:

  ```sh
  ./run clear_cache
  ```

In some cases you may also want to clear out datasets downloaded by applications to the `data` folder:

  ```sh
  rm -rf ./data
  ```

Note that many applications supply custom container environments with build and runtime dependencies.
Failing to clean the build cache between different applications may result in unexpected behavior where build
tools or libraries appear to be broken or missing. Clearing the build cache is a good first check to address those issues.

# Contributing

The goal of this repository is to allow engineering teams to easily contribute and share new functionalities
and to demonstrate applications. Please review the [Contributing Guidelines](./CONTRIBUTING.md) for more information.

# Glossary

Many applications use the following keyword definitions in their README descriptions:

- `<HOLOHUB_SOURCE_DIR>` : Path to the source directory
- `<HOLOHUB_BUILD_DIR>` : Path to the build directory
- `<HOLOSCAN_INSTALL_DIR>` : Path to the installation directory of Holoscan SDK
- `<DATA_DIR>` : Path to the top level directory containing the datasets for the reference applications
- `<MODEL_DIR>` : Path to the directory containing the inference model(s)

# More Information

Refer to additional documentation:
- [Contributing Guide](/CONTRIBUTING.md)
- [Developer Guide](/doc/developer.md)
- [Release Discussion](/doc/release.md)

You can find additional information on Holoscan SDK at:
- [Holoscan GitHub organization](https://github.com/nvidia-holoscan)
- [Holoscan SDK repository](https://github.com/nvidia-holoscan/holoscan-sdk)
- [Holoscan-GPT](https://chatgpt.com/g/g-M6hMJimGa-holochatgpt) (requires an OpenAI account)
- [Holoscan Support Forum](https://forums.developer.nvidia.com/c/healthcare/holoscan-sdk/320/all)
