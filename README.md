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
HoloHub has been tested and is known to run on Ubuntu 20.04.
Other versions of Ubuntu or OS distributions may result in build and/or runtime issues.

1. Clone this repository.

2. Choose to build Holohub using development container or using a local environment.

***Note:*** The default [NGC's Holoscan SDK container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan) version for the [Container build](#container-build) process is updated to **`v1.0`**, but if you would like to use an older version of the SDK, simply specify your own `--base_img` (see *Advanced build options* below).
### Container build

***Prerequisites***
To build and run Holohub in a containerized environment you will need:
  - the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) v1.12.2+
  - [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository), including the buildx plugin (`docker-buildx-plugin`)

***Building dev container:*** Run the following command from the holohub directory to build the development container.

```bash
  ./dev_container build
```

***Note:*** The development container script ```dev_container``` will by default use [NGC's Holoscan SDK container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan) for the local GPU configuration by detecting if the system is using an iGPU (integrated GPU) or a dGPU (discrete GPU).

For example, on an x86_64 system with dGPU, ```docker images``` will list the following new images after ```./dev_container build``` has completed.
```bash
user@ubuntu-20-04:/media/data/github/holohub$ docker images
REPOSITORY                               TAG           IMAGE ID       CREATED         SIZE
holohub                                  ngc-v0.6.0-dgpu   b6d86bccdcac   9 seconds ago   8.47GB
nvcr.io/nvidia/clara-holoscan/holoscan   v0.6.0-dgpu       1b4df7733d5b   5 weeks ago     8.04GB
```

***Advanced build options***

1. Custom base image

It is possible to configure a custom base image for building the Holohub container. E.g., if you built Holoscan SDK locally and want to use the locally built container as the base image, use the following command:

```bash
  ./dev_container build --base_img holoscan-sdk-dev:latest --img holohub:sdk-dev-latest
```

where ```--base_img```  is used to configure the base container image and ```--img``` defines the fully qualified image name.

After ```./dev_container build``` has completed ```docker images``` will list the following new image

```bash
user@ubuntu-20-04:/media/data/github/holohub$ docker images
REPOSITORY                               TAG               IMAGE ID       CREATED          SIZE
holohub                                  sdk-dev-latest    cb0231f77856   54 seconds ago   8.22GB
```


2. Using custom Dockerfile

```bash
  ./dev_container build  --docker_file <path_to_dockerfile>  --img holohub-debug:local-sdk-v0.6.0
```

where ```--docker_file```  is the path to the container's Dockerfile and ```--img``` defines the fully qualified image name.

***Note***:  To print the values for base image, docker file, gpu type and output image name use ```--verbose```.

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


***Launching dev container:***  Run the following command from the holohub directory to launch the default development container built using Holoscan SDK's container from ngc for the local GPU.

```bash
  ./dev_container launch
```

The launch script will also inspect for available video devices (V4L2, AJA capture boards, Deltacast capture boards) and the presence of Deltacast's Videomaster SDK and map it into the development container.

***Advanced launch options***

1. For Holohub image built with locally built Holoscan SDK container

```bash
  ./dev_container launch --img holohub:local-sdk-latest --local_sdk_root <path_to_holoscan_sdk>
```

2. Launch custom Holohub container with fully qualified name, e.g. "holohub:ngc-sdk-sample-app"

```bash
  ./dev_container launch --img holohub:ngc-sdk-sample-app
```

3. Enable X11 forwarding of graphical HoloHub applications over SSH

```bash
  ./dev_container launch --ssh_x11
```

4. Support Nsight Systems profiling in container

```bash
  ./dev_container launch --nsys_profile
```

***Note***:  To print the values passed to the docker run command, add the ```--verbose``` option to the launch command.

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

Please note that the values of some of the variables will vary depending on configured options, iGPU or dGPU, availability of devices for video capture, ...


From within the container build the Holohub apps as explained in section [Building HoloHub](#building-holohub).

The development container has been tested on the following platforms: x86_x64 workstation with multiple RTX GPUs, Clara AGX Dev Kit (dGPU mode), IGX Orin Dev Kit (dGPU and iGPU mode), AGX Orin Dev Kit (iGPU).


***Notes for AGX Orin Dev Kit***:

(1) On AGX Orin Dev Kit the launch script will add ```--privileged``` and ```--group-add video``` to the docker run command for the Holohub sample apps to work. Please also make sure that the current user is member of the group video.

(2) When building Holoscan SDK on AGX Orin Dev Kit from source please add the option  ```--cudaarchs all``` to the ```./run build``` command to include support for AGX Orin's iGPU.


### Local build

Refer to the [Holoscan SDK README](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/README.md) for ways to install Holoscan SDK in local environment: Debian package, Python wheels or from source.

Install the package dependencies for Holohub on your host system. The easiest way to make sure the minimal package dependencies is to use the run script from the top level directory.

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

## Advanced launch options

1. Pass additional arguments to the application command

```bash
  ./run launch endoscopy_tool_tracking python --extra_args '-r visualizer'
```

2. Profile using Nsight Systems

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

The goal of Holohub is to allow engineering teams to easily contribute and share new functionalities
and demonstration applications. Please review the [CONTRIBUTING.md file](https://github.com/nvidia-holoscan/holohub/blob/main/CONTRIBUTING.md) for more information on how to contribute.

# Useful Links

- [Holoscan GitHub organization](https://github.com/nvidia-holoscan)
- [Holoscan SDK repository](https://github.com/nvidia-holoscan/holoscan-sdk)
