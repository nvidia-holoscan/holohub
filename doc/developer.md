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

Install the package dependencies for HoloHub on your host system. The easiest way to make sure the minimal package dependencies is to use the `./holohub` script from the top level directory.

```bash
  ./holohub setup  # sudo privileges may be required
```

If you prefer you can also install the dependencies manually, typically including the following:
- [CMake](https://www.cmake.org): 3.24.0+
- Python interpreter: 3.9 to 3.12
- Python dev: 3.9 to 3.12 (matching version of the interpreter)
- ffmpeg runtime
- [ngc-cli](https://ngc.nvidia.com/setup/installers/cli)
- wget
- CUDA Toolkit: 12.6
- libcudnn9-cuda-12
- libcudnn9-dev-cuda-12
- libnvinfer-dev
- libnvinfer-plugin-dev
- libnvonnxparsers-dev

Visit the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html) for the latest
details on dependency versions and custom installation.

*Note: the `./holohub` script setup installs the minimal set of dependencies required to run the sample applications. Other applications might require more dependencies. Please refer to the README of each application for more information.*

## Advanced Build Options (Container)

### View All Options

Run the following to view all build options available for the HoloHub container script:
```sh
$ ./holohub build-container --help
```

### Custom Base Image

You may configure a custom base image for building the HoloHub container. For instance, if you want to use a local Holoscan container as the base image, use the following command:

```bash
  ./holohub build-container --base-img holoscan-sdk-build-x86_64:latest --img holohub:sdk-dev-latest
```

The command above uses the following arguments:
- `--base-img`  is used to configure the base container image;
- `--img` defines the fully qualified name of the image output by `./holohub`.

After ```./holohub build-container``` has completed ```docker images``` will list the new image:

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
$ ./holohub build-container --docker-file <path_to_dockerfile>  --img holohub-debug:latest
```

Where:
- `--docker-file`  is the path to the container's Dockerfile;
- `--img` defines the fully qualified image name.

### Build with Verbose Output

To print the values for base image, Dockerfile, GPU type, and output image name, use ```--verbose```.

For example, on an x86_64 system with dGPU, the default build command will print the following values when using the ```--verbose``` option.

```bash
user@ubuntu-20-04:/media/data/github/holohub$ ./holohub build-container --verbose
Build (HOLOHUB_ROOT:/media/data/github/holohub)...
Build (gpu_type_type:dgpu)...
Build (base_img:nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu)...
Build (docker_file_path:/media/data/github/holohub/Dockerfile)...
Build (img:holohub:ngc-v0.6.0-dgpu)...
....
```

## Advanced Launch Options (Container)

### View All Options

Run the command below to view all available launch options in the `holohub` script:

```sh
$ ./holohub run-container --help
```

### Build and Launch a Local Holoscan SDK Container

To use a HoloHub container image built with a local Holoscan SDK container:

```bash
$ ./holohub run-container --img holohub:local-sdk-latest --local-sdk-root <path_to_holoscan_sdk>
```

### Launch a Named HoloHub Container

To launch custom HoloHub container with fully qualified name, e.g. "holohub:ngc-sdk-sample-app"

```bash
$ ./holohub run-container --img holohub:ngc-sdk-sample-app --no-docker-build
```

### Forward X11 Graphics Over SSH

```bash
  ./holohub run-container --ssh-x11
```

### Support Nsight Systems profiling in the HoloHub Container

```bash
  ./holohub run-container --nsys-profile
```

## Advanced Options for Building Applications

### View Build Options

```sh
./holohub build --help
```

### Local SDK Path

If you have an installation of the Holoscan SDK which is not in a standard path, you may want to provide the root directory of your Holoscan SDK installation.

```bash
  ./holohub build --configure-args="-Dholoscan_DIR=/path/to/holoscan/install/lib/cmake/holoscan"
```

### Building a Specific Application

By default HoloHub builds all the sample applications that are maintained with the SDK. You can build specific applications by the name of the directory.

```bash
  ./holohub build <application>
```

For example:

```bash
  ./holohub build endoscopy_tool_tracking
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

The `./holohub` script creates a `data` subdirectory to store the downloaded HoloHub data.
This directory is noted `HOLOHUB_DATA_DIR/holohub_data_dir` in the documentation, READMEs and metadata files.

## Advanced Options for Running Applications

### Pass additional arguments to the application command

```bash
  ./holohub run endoscopy_tool_tracking --language=python --run-args='-r visualizer'
```

### Profile using Nsight Systems

```bash
  ./holohub run endoscopy_tool_tracking --language=python --nsys-profile
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
