# Welcome to HoloHub!

Holohub is a central repository for users and developers of extensions and applications for the Holoscan Platform to share reusable components and sample applications.

# Table of Contents
- [Prerequisites](#prerequisites)
- [Building HoloHub](#building-holohub)
- [Running applications](#running-applications)
- [Contributing to HoloHub](#contributing-to-holohub)
- [Useful Links](#useful-links)

# Prerequisites
HoloHub is based on [Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk).
HoloHub has been tested and is known to run on Ubuntu 20.04. Other versions of Ubuntu or OS distributions may result in build and/or runtime issues.


1. Clone this repository.

2. Choose to build Holohub sample apps using development container or bare metal.

**Container build**

***Build container:*** Run the following command from the holohub directory to build the development container.

```bash
  ./dev_container build_image
```

***Note:*** The development container script ```dev_container``` will by default use [NGC's Holoscan SDK container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan) for the local GPU configuration by detecting if the system is using an iGPU (integrated GPU) or a dGPU (discrete GPU). 


It is possible to configure a locally built Holoscan SDK container by passing the option ```--base_image from_source``` to the build_image or launch command. When using the ```--base_image from_source```  option please make sure that you have checked out holoscan-sdk into the same directory as holohub and that you have built holoscan SDK as described in the [Holoscan SDK README: Build using the run script](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/README.md#recommended-using-the-run-script).


***Launch container:***  Run the following command from the holohub directory to launch the development container.

```bash
  ./dev_container launch
```

From within the container build the Holohub apps as explained in section [Building HoloHub](#building-holohub).

The development container has been tested on the following platforms: x86_x64 workstation with multiple RTX GPUs, Clara AGX Dev Kit (dGPU mode), IGX Orin Dev Kit (dGPU mode), AGX Orin Dev Kit (iGPU).

The launch script will also inspect for available video devices (V4L2, AJA capture boards, Deltacast capture boards) and presence of Deltacast's Videomaster SDK and map it into the development container.

***Notes for AGX Orin Dev Kit***: 

(1) On AGX Orin Dev Kit the launch script will add ```--privileged``` and ```--group-add video``` to the docker run command for the Holohub sample apps to work. Please also make sure that the current user is member of the group video.  

(2) When building Holoscan SDK on AGX Orin Dev Kit from source please add the option  ```--cudaarchs all``` to the ```./run build``` command to include support for AGX Orin's iGPU.


**Bare metal build** 

Refer to the [Holoscan SDK README](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/README.md) for ways to install Holoscan SDK bare metal: Debian package, Python wheels or from source.

Install the package dependencies for Holohub on your host system. The easiest way to make sure the minimal package dependencies is to use the run script from the top level directory.

```bash
  # if sudo is available
  sudo ./run setup
```

If you prefer you can also install the dependencies manually:
- [CMake](https://www.cmake.org): 3.20.1+
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

# Building HoloHub 

While not all applications requires building HoloHub, the current build system automatically manages dependencies (applications/operators) and also downloads and convert datasets at build time. HoloHub provides a convenient run script to build and run applications (you can run `./run -h` for information about the available commands).

You can refer to the README of each application/operator if you prefer to build/run them manually.

Make sure you have installed the [prerequisites](#prerequisites) before attempting to build HoloHub.
## Building Sample applications
Sample applications based on the Holoscan Platform may be found under the [Applications](./applications/) directory. Sample applications are a subset of the HoloHub applications and are maintained by Holoscan SDK developers to provide a demonstration of the SDK capabilities.

To build sample applications, make sure you have install the prerequisites and setup your NGC credentials then run:

```bash
  ./run build
```

Alternatively if you have an installation of the Holoscan SDK which is not in a standard path, you may want to provide the root directory of your Holoscan SDK installation.

```bash
  ./run build --sdk <path to the Holoscan SDK installation directory>
```

*Note that the run script creates a `data` directory to put the downloaded Holohub data where the run script is located.
This directory is noted HOLOHUB_DATA_DIR/holohub_data_dir in the documentation, READMEs and metadata files.*

When the build is successful you can [run the sample applications](#running-applications)

## Building a specific application
By default HoloHub builds the sample applications that are maintained with the SDK, but you can build specific applications by the name of the directory.

```bash
  ./run build <application>
```

for example:

```bash
  ./run build endoscopy_tool_tracking
```

Note that CMake with build the application in the directory specified if there are multiple languages, the script will attempt to build all of them.


## Building application or operator manually
If you prefer to build applications and operator manually you can follow the steps below.

```bash
# Export cuda (in case it's not already in the path)
export PATH=$PATH:/usr/local/cuda/bin

# Configure Holohub with CMake
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

# Running applications
To list all available applications you can run the following command:
```bash
  ./run list
```

Then you can run the application using
```bash
  ./run launch <application> <language>
```

For example to run the tool tracking endoscopy application in C++
```bash
  ./run launch endoscopy_tool_tracking cpp
```

and to run the same application in python:
```bash
  ./run launch endoscopy_tool_tracking python
```

The run script reads the "run" command from the metadata.json file for a given application and runs from the "workdir" directory.
Make sure you build the application (if applicable) before running it.

# Contributing to HoloHub

The goal of Holohub is to allow engineering teams to easily contribute and share new functionalities
and demonstration applications. Please review the [CONTRIBUTING.md file](https://github.com/nvidia-holoscan/holohub/blob/main/CONTRIBUTING.md) for more information on how to contribute.

# Useful Links

- [Holoscan GitHub organization](https://github.com/nvidia-holoscan)
- [Holoscan SDK repository](https://github.com/nvidia-holoscan/holoscan-sdk)
