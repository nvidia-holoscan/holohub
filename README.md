# Holoscan Camera

Camera source components for Holoscan SDK 5.x EA.

This is a Holoscan Module: a self-contained, redistributable C++ library that
extends [Holoscan SDK](https://developer.nvidia.com/holoscan-sdk) with reusable
operators under the `holoscan::holoscan_camera` namespace.

## Holoscan SDK 5.0 Early Access Setup

Before building Holoscan Camera, build Holoscan SDK 5.0 RC5 from source.
See the [SDK 5.0 RC5 release notes](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v5.0.0-rc5/RELEASE_NOTES.md)
for release details.

Holoscan Camera 0.4.0 requires a post-EA2 SDK revision. The released SDK 5.0 EA2 sources
(`v5.0.0.3` internally or `v5.0.0-ea2` publicly) do not support the SIPL schema
generation and shared-pointer constructor captures used here. The SDK package
version remains `5.0.0`, so a successful package-version check alone does not
establish compatibility. See the [0.4.0 migration notes](RELEASE_NOTES.md#040).

Set `HOLOSCAN_SDK_INSTALL_DIR` explicitly to the resulting SDK installation
directory. The installation must include `holoscan::sensor_io`, which this
module requires. See [SDK installation troubleshooting](#sdk-installation-troubleshooting)
if CMake rejects the installation.

The SDK build provides the base container image used by `./holoscan_camera`:
`holoscan-sdk-build-x86_64:latest` on x86_64 and
`holoscan-sdk-build-aarch64:latest` on SBSA.

Build the SDK and set the installation path:

```bash
# Start from your Holoscan SDK 5.x working directory. The run script lives in public/,
# not at the repository root.
cd <path/to/holoscan-sdk>/public

# Build the SDK base container environment and installation
./run build

# Set the installation path and ensure it is populated
export HOLOSCAN_SDK_INSTALL_DIR="$(pwd)/$(./run get_install_dir)"

# Verify the installation path looks like: /path/to/holoscan-5x/install-<arch>
echo ${HOLOSCAN_SDK_INSTALL_DIR}

# Return to the Holoscan Camera project
cd -
```

## Operators and documentation

| Operator | Purpose | Documentation |
| --- | --- | --- |
| `V4l2CaptureOp` | Capture packed YUYV frames from Linux V4L2 devices into host, pinned-host, or CUDA device memory | [Usage and interface](operators/v4l2_capture_op/README.md) |
| `SIPLCaptureOp` | Capture RAW10/NV12 frames from NvSIPL-managed cameras on Jetson/IGX platforms (aarch64 only) | [Usage and interface](operators/sipl_capture_op/README.md) |

- [Release Notes and Known Issues](RELEASE_NOTES.md)
- [V4L2 benchmarks and qualification results](docs/v4l2_capture_op_benchmarks.md)
- [SIPL benchmarks and qualification results](docs/sipl_capture_op_benchmarks.md)
- [CI and developer checks](ci/README.md)

## Quick Start

Run these commands from the repository root after completing SDK setup:

```bash
# Build and validate the V4L2 example graph without a camera.
./holoscan_camera run holoscan_camera_v4l2

# Run the reference application's tests.
./holoscan_camera test holoscan_camera_v4l2 --language cpp

# Build the Debian package.
./holoscan_camera package holoscan-camera --pkg-generator DEB
```

For device access, capture options, C++ integration, and frame placement, see the
[V4L2 operator guide](operators/v4l2_capture_op/README.md).

`SIPLCaptureOp` targets Jetson/IGX (aarch64) and needs a mapped NvSIPL-managed
camera; see [sipl_frame_saver](applications/sipl_frame_saver) and
[sipl_stereo_monitor](applications/sipl_stereo_monitor) for reference
applications, and the [SIPL operator guide](operators/sipl_capture_op/README.md)
for supported JetPack/L4T versions and usage.

## Building Without HoloHub CLI

| Requirement | Version |
| --- | --- |
| Holoscan SDK | >= 5.0.0 EA |
| CUDA Toolkit | 13.x |
| CMake | >= 4.0 |
| C++ compiler | C++20 |

```bash
cmake -S . -B build -G Ninja \
    -Dholoscan_ROOT="$HOLOSCAN_SDK_INSTALL_DIR" \
    -DBUILD_ALL=ON \
    -DHOLOSCAN_CAMERA_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure -LE gpu
```

`-LE gpu` drops the device-resident placement check, which is the one test that
needs a visible CUDA device. Drop the exclusion on a runner that has one.

## Language Support

Holoscan SDK 5.0 EA support for this module is C++ only. Non-C++ module
surfaces have been removed until SDK 5.0 module support expands.

## FAQ + Troubleshooting

### Integrating with custom Holoscan SDK 5.0 development

Use the local SDK workflow documented above to integrate and develop Holoscan Camera changes against Holoscan SDK 5.0 development:

1. Develop and test changes in the Holoscan SDK 5.0 project
2. Update the Holoscan SDK 5.0 installation folder to latest development
3. Rebuild Holoscan Camera against the local SDK installation as documented above

### SDK installation troubleshooting

The wrapper can auto-detect an SDK installation under `../holoscan-sdk`, but
that check does not verify whether it includes `holoscan::sensor_io`. Set
`HOLOSCAN_SDK_INSTALL_DIR` explicitly to select the SDK installation you built.

If CMake finds the SDK package but reports `holoscan_FOUND` as `FALSE`, the
installation may be missing the required `sensor_io` component. Check for
`holoscan-sensor-io-targets.cmake` in the SDK installation. If it is missing,
rebuild and install Holoscan SDK 5.0 RC5, then update
`HOLOSCAN_SDK_INSTALL_DIR` and configure Holoscan Camera again.

### Build Failure: `[holoscan_camera] Invalid Holoscan SDK installation: holoscan-sdk/install-cu12-x86_64`

Holoscan Camera depends on Holoscan SDK 5.0 and its CUDA 13 dependency. Earlier Holoscan installations are not
compatible. Please clear any existing build and installation folders, check out the Holoscan SDK 5.0 RC5 source,
and try again.

## License

Apache-2.0 - see [LICENSE](LICENSE).
