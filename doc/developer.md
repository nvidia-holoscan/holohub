# HoloHub Developer Reference

In this guide we aim to document advanced workflows to aid developers in working with HoloHub,
and to support advanced project use cases.

## Table of Contents

- [The `./holohub` Script and the Holoscan CLI](#the-holohub-script-and-the-holoscan-cli)
- [Native Build](#native-build)
- [Advanced Container Build Options](#advanced-build-options-container)
- [Advanced Container Launch Options](#advanced-launch-options-container)
- [Advanced Options for Building Applications](#advanced-options-for-building-applications)
- [Advanced Options for Running Applications](#advanced-options-for-running-applications)

## The `./holohub` Script and the Holoscan CLI

The `./holohub` script is a thin wrapper around the standalone
[holoscan-cli](https://github.com/nvidia-holoscan/holoscan-cli) package. On
first use it selects a Python environment for the CLI — typically a
wrapper-managed venv under `~/.local/share/holoscan-cli/venv` (requires
`python3-venv`) — and installs the package there, so the first run needs
network access. Run `./holohub` as your normal user; **no sudo needed** —
`setup` elevates the individual system operations itself.

See [utilities/cli/README.md](../utilities/cli/README.md) for the full
environment-selection order and the `HOLOSCAN_CLI_*` variables that control
it (interpreter, venv location, source checkout, install arguments, version
pin, and root system-install policy). `./holohub env-info` reports which
environment was selected. To remove the managed environment entirely:

```bash
rm -rf "${XDG_DATA_HOME:-$HOME/.local/share}/holoscan-cli/venv"
```

The next ordinary host invocation creates a new managed environment.

## Native Build

### Software Prerequisites (Native)

Refer to the [Holoscan SDK README](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/README.md) for ways to install Holoscan SDK in local environment: Debian package, Python wheels or from source.

Install the package dependencies for HoloHub on your host system. The easiest way to make sure the minimal package dependencies is to use the `./holohub` script from the top level directory.

```bash
  ./holohub setup  # run as your normal user; individual steps elevate with sudo as needed
```

If you prefer you can also install the dependencies manually, typically including the following:

- [CMake](https://www.cmake.org): 3.24.0+
- Python interpreter: 3.10 to 3.13
- Python dev: 3.10 to 3.13 (matching version of the interpreter)
- ffmpeg runtime
- [ngc-cli](https://ngc.nvidia.com/setup/installers/cli)
- wget
- The CUDA Toolkit, cuDNN, and TensorRT versions required by the installed
  Holoscan SDK and selected platform

Visit the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation) for the latest
details on dependency versions and custom installation.

*Note: the `./holohub` script setup installs the minimal set of dependencies required to run the sample applications. Other applications might require more dependencies. Please refer to the README of each application for more information.*

## Advanced Build Options (Container)

### View All Build Options

Run the following to view all build options available for the HoloHub container script:

```sh
./holohub build-container --help
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
./holohub build-container --docker-file <path_to_dockerfile>  --img holohub-debug:latest
```

Where:

- `--docker-file`  is the path to the container's Dockerfile;
- `--img` defines the fully qualified image name.

### Inspect the Container Build

Use `--dryrun --verbose` to resolve the base image, Dockerfile, GPU type,
build arguments, and output tags without building an image:

```bash
./holohub build-container --dryrun --verbose
```

The preview prints the exact `docker build` command. Values depend on the
wrapper's configured SDK version, detected GPU type, selected CUDA major
version, current branch, and commit.

## Advanced Launch Options (Container)

### View All Launch Options

Run the command below to view all available launch options in the `holohub` script:

```sh
./holohub run-container --help
```

### Build and Launch a Local Holoscan SDK Container

To use a HoloHub container image built with a local Holoscan SDK container:

```bash
./holohub run-container --img holohub:local-sdk-latest --local-sdk-root <path_to_holoscan_sdk>
```

where `<path_to_holoscan_sdk>` is the path to the Holoscan SDK root directory containing the build directory.
Please refer to the [Holoscan SDK Developer Guide](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/DEVELOP.md) for more details on how to build the Holoscan SDK from source.

In the container, to verify the build directory (with a python build) is mounted correctly, run the following command:

```bash
python -c "import holoscan; print(holoscan.__file__)"
```

The output should be something like:

```bash
/workspace/holoscan-sdk/build-x86_64/python/lib/holoscan/__init__.py
```

If Python supported is not enabled, `/workspace/holoscan-sdk` can be manually inspected to confirm the mount.
The directory should contain a non-empty `build-<arch>-<gpu_type>` or `install-<arch>-<gpu_type>` directory.

### Launch a Named HoloHub Container

To launch custom HoloHub container with fully qualified name, e.g. "holohub:ngc-sdk-sample-app"

```bash
./holohub run-container --img holohub:ngc-sdk-sample-app --no-docker-build
```

### Forward Graphics From Containers

HoloHub automatically forwards X11 and Wayland displays when `DISPLAY` or
`WAYLAND_DISPLAY` is set on the host.

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

To build against a Holoscan SDK source/build tree on the host, use
`--local-sdk-root`:

```bash
./holohub build <project> \
  --local-sdk-root /path/to/holoscan-sdk
```

If a nonstandard SDK install is already visible inside the selected container
or native environment, pass its CMake package directory explicitly:

```bash
./holohub build <project> \
  --configure-args="-Dholoscan_DIR=/path/to/holoscan/install/lib/cmake/holoscan"
```

### Building a Specific Application

`./holohub build` requires a discovered project name and builds that project
plus its declared dependencies. Use `./holohub list` to see valid names.

```bash
  ./holohub build <application>
```

For example:

```bash
  ./holohub build endoscopy_tool_tracking
```

If a project has multiple language implementations, the CLI reports its
selection and defaults to Python when available. Pass `--language cpp` or
`--language python` to select an implementation explicitly.

### Building an application or operator manually

Prefer entering the project's development container before invoking CMake
directly. This keeps host dependencies and generated files aligned with the
standard CLI build environment:

```bash
./holohub run-container <project>
```

Inside the container, configure and build with commands such as:

```bash
# Export CUDA if it is not already in PATH.
export PATH="$PATH:/usr/local/cuda/bin"

# Configure HoloHub with CMake (example application)
cmake -S /workspace/holohub \
  -B build \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DHOLOHUB_DATA_DIR="$(pwd)/data" \
  -DAPP_endoscopy_tool_tracking=ON

# Build the application(s)
cmake --build build
```

Direct host CMake builds are the native-build path and require the host setup
described above.

## Additional Build Notes

While not all applications requires building HoloHub, the current build system automatically manages dependencies (applications/operators) and also downloads and converts datasets at build time.

You can refer to the README of each application/operator if you prefer to build/run them manually.

The CLI creates a `data` subdirectory for downloaded HoloHub data. CMake uses
`HOLOHUB_DATA_DIR`; application metadata refers to the same location through
the `<holohub_data_dir>` placeholder. At runtime, the CLI also exports
`HOLOSCAN_CLI_DATA_PATH` and `HOLOSCAN_INPUT_PATH`.

## Advanced Options for Running Applications

### Pass additional arguments to the application command

```bash
  ./holohub run endoscopy_tool_tracking --language=python --run-args='-r visualizer'
```

### Profile using Nsight Systems

For example, to profile `endoscopy_tool_tracking` using Nsight Systems:

First config the app's replay count to 10 frames:

```diff
--- a/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.yaml
+++ b/applications/endoscopy_tool_tracking/python/endoscopy_tool_tracking.yaml
@@ -81,7 +81,7 @@ replayer:
   frame_rate: 0   # as specified in timestamps
   repeat: true    # default: false
   realtime: true  # default: true
-  count: 0        # default: 0 (no frame count restriction)
+  count: 10

```

Then run the app with `--nsys-profile` option:

```bash
  ./holohub run endoscopy_tool_tracking --language=python --nsys-profile
```

This will create a Nsight Systems report file in the application working directory. Information on the generated report file is printed on the end of the application log:

```text
Generating '/tmp/nsys-report-bcd8.qdstrm'
[1/1] [========================100%] report8.nsys-rep
Generated:
    /workspace/holohub/build/endoscopy_tool_tracking/report8.nsys-rep
```

This file can be loaded and visualized with the Nsight Systems UI application:

```bash
  nsys-ui /workspace/holohub/build/endoscopy_tool_tracking/report8.nsys-rep
```
