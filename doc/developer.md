# HoloHub Developer Reference

In this guide we aim to document advanced workflows to aid developers in working with HoloHub,
and to support advanced project use cases.

# Table of Contents
- [Native Build](#native-build)
- [Advanced Container Build Options](#advanced-build-options-container)
- [Advanced Container Launch Options](#advanced-launch-options-container)
- [Advanced Options for Building Applications](#advanced-options-for-building-applications)
- [Advanced Options for Running Applications](#advanced-options-for-running-applications)

## Native Build

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

## Advanced Build Options (Container)

### View All Options

Run the following to view all build options available for the HoloHub container script:
```sh
$ ./dev_container help build
```

### Custom Base Image

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

### Using a Custom Dockerfile

Several HoloHub applications use a custom Dockerfile to alter or extend the default HoloHub container. Use the following command to build from a custom Dockerfile:

```bash
$ ./dev_container build  --docker_file <path_to_dockerfile>  --img holohub-debug:local-sdk-v0.6.0
```

Where:
- `--docker_file`  is the path to the container's Dockerfile;
- `--img` defines the fully qualified image name.

### Build with Verbose Output

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

## Advanced Launch Options (Container)

### View All Options

Run the command below to view all available launch options in the `dev_container` script:

```sh
$ ./dev_container help launch
```

### Launch a Local Holoscan SDK Container

To use a HoloHub container image built with a local Holoscan SDK container:

```bash
$ ./dev_container launch --img holohub:local-sdk-latest --local_sdk_root <path_to_holoscan_sdk>
```

### Launch a Named HoloHub Container

To launch custom HoloHub container with fully qualified name, e.g. "holohub:ngc-sdk-sample-app"

```bash
$ ./dev_container launch --img holohub:ngc-sdk-sample-app
```

### Forward X11 Graphics Over SSH

```bash
  ./dev_container launch --ssh_x11
```

### Support Nsight Systems profiling in the HoloHub Container

```bash
  ./dev_container launch --nsys_profile
```

### Print Verbose Output

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

## Advanced Options for Building Applications

### View Build Options

```sh
./run -h
```

### Local SDK Path

If you have an installation of the Holoscan SDK which is not in a standard path, you may want to provide the root directory of your Holoscan SDK installation.

```bash
  ./run build --sdk <path to the Holoscan SDK installation directory>
```

### Building a Specific Application

By default HoloHub builds all the sample applications that are maintained with the SDK. You can build specific applications by the name of the directory.

```bash
  ./run build <application>
```

For example:

```bash
  ./run build endoscopy_tool_tracking
```

Note that CMake will build the application in the directory specified. If there are multiple languages, the script will attempt to build all of them.


### Building application or operator manually

If you prefer to build applications and operator manually you can follow the steps below.

```bash
# Export cuda (in case it's not already in the path)
export PATH=$PATH:/usr/local/cuda/bin

# Configure HoloHub with CMake
cmake -S <path_to_holohub_source>            # Source directory
      -B build                               # Build directory
      -DPython3_EXECUTABLE=/usr/bin/python3  # Specifies the python executable for CMake to find the correct version
      -DHOLOHUB_DATA_DIR=$(pwd)/data         # Specifies the data directory
      -DAPP_<name_of_the_application>=1      # Specified the application to build


# Build the application(s)
cmake --build build
```

## Additional Build Notes

While not all applications requires building HoloHub, the current build system automatically manages dependencies (applications/operators) and also downloads and converts datasets at build time.

You can refer to the README of each application/operator if you prefer to build/run them manually.

The run script creates a `data` subdirectory to store the downloaded HoloHub data.
This directory is noted `HOLOHUB_DATA_DIR/holohub_data_dir` in the documentation, READMEs and metadata files.

## Advanced Options for Running Applications

### Pass additional arguments to the application command

```bash
  ./run launch endoscopy_tool_tracking python --extra_args '-r visualizer'
```

### Profile using Nsight Systems

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
