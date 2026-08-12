# Developer Guide

## Repository layout

```text
holoscan-5.0-early-access-samples/
├── applications/
│   └── v4l2_depth/
│       ├── cpp/
│       │   ├── metadata.json
│       │   └── v4l2_depth.cu
│       ├── docs/v4l2_depth_loopback.png
│       ├── scripts/
│       │   ├── feed_v4l2_loopback.py
│       │   ├── prepare_model.sh
│       │   └── run_v4l2_depth_loopback.py
│       └── Dockerfile
├── operators/
│   ├── bgra_to_planar_tensor/
│   ├── cuda_compositor/
│   ├── depth_colorizer/
│   ├── frame_skipper/
│   ├── tensorrt_inference/
│   ├── v4l2_depth_common/
│   ├── v4l2_source/
│   ├── x11_display/
│   └── yuyv_to_bgra/
├── cmake/
│   ├── HoloHubConfigHelpers.cmake
│   └── container.ctest
├── tests/
├── Dockerfile
├── metadata.json
└── holohub
```

Keep the top level collection-oriented. Future applications and operators should be sibling
projects with independent metadata and CMake entry points.

## SDK API baseline

Treat the `examples` directory in a Holoscan SDK 5.0 checkout as the source of truth for the
evolving 5.0 EA API. This project follows its typed graph pattern:

- Operators derive from `holoscan::Operator<>`.
- Ports are typed `holoscan::Input<T>` and `holoscan::Output<T>` members.
- `setup()` declares port names, queue depths, and emission bounds.
- Operators that need a specific temporal trigger override `contract()` with `OnClock` or
  `OnEach`. The compositor intentionally uses the inputful empty contract, which lowers to
  any-input activation so either a base frame or an overlay update can select it.
- `compute()` returns `holoscan::expected<void, holoscan::Error>`.
- Applications construct a `holoscan::Graph`, compile an immutable execution plan, and start it
  with `run_async()`.

The project uses C++20 and requires Holoscan 5.0.0.

The vision sample additionally follows the SDK 5.x Tensor loan protocol:

- Tensor ports declare memory kind, dtype, rank, and bounded output allocations.
- CUDA output ports receive explicit `bind_tensor_output_device()` placements at compile time.
- Producers allocate a loan, obtain a stream-ordered writer, enqueue work, commit, and emit.
- Consumers receive tensors whose readiness is pre-ordered on the operator runtime stream.
- Pass-through operators republish the received `Sample<Tensor>` and do not claim allocation
  authority.
- Host/device transfers are explicit because the SDK 5.x runtime does not insert conversion
  bridges.

The eight operators required by `v4l2_depth` are independent projects with their own metadata and
CMake targets, so each appears under `./holohub list` and can be built independently. Each
operator's header and source live directly in its project directory; there is no extra
`include/v4l2_depth_operators` or `src` nesting. `v4l2_depth_common` is an internal interface
library shared by those operators, not a ninth discoverable operator.

## CMake discovery

Component selection follows HoloHub conventions:

```cmake
add_holohub_application(v4l2_depth
    DEPENDS
        OPERATORS
            bgra_to_planar_tensor
            cuda_compositor
            depth_colorizer
            frame_skipper
            tensorrt_inference
            v4l2_source
            x11_display
            yuyv_to_bgra
)

add_holohub_operator(bgra_to_planar_tensor)
# Register the other seven operator projects the same way.
```

These helpers define `APP_<name>` and `OP_<name>`. Applications are discovered before operators so
application dependencies can enable the required operator option before the operator pass.
`BUILD_ALL` defaults to `OFF`; the CLI enables the requested project and its dependencies.

For an operator-only build:

```bash
./holohub build frame_skipper \
    --local \
    --configure-args=-DBUILD_ALL=OFF
```

## SDK and container selection

Build and install the SDK before using this project. Run the build from the SDK checkout root,
which contains the `run` script.

```bash
cd /path/to/holoscan-sdk
./run build \
    --build-python false \
    --build-benchmarks false

export HOLOSCAN_SDK_INSTALL_DIR=/path/to/holoscan-sdk/install-$(uname -m)
./holohub build v4l2_depth
```

`--local-sdk-root` may identify the installation tree directly or a parent containing the
architecture-specific tree. Holoscan CLI mounts the selected installation at
`/workspace/holoscan-sdk` for container builds.

`HOLOSCAN_SDK_INSTALL_DIR` and `HOLOSCAN_SDK_ROOT` remain supported for commands without the CLI
flag. A sibling `../holoscan-sdk/install-$(uname -m)` tree is discovered automatically.
`HOLOSCAN_SDK_INSTALL_DIR` must name a direct SDK installation. `HOLOSCAN_SDK_ROOT` may name either
a direct installation or a parent containing `install-x86_64` or `install-aarch64`.

When neither variable is set, the wrapper leaves SDK resolution at the conventional
`/opt/nvidia/holoscan` installation.

The wrapper also supplies these project defaults:

- `HOLOSCAN_CLI_BASE_IMAGE=holoscan-sdk-build-<architecture>:latest`
- `HOLOSCAN_CLI_CTEST_SCRIPT=cmake/container.ctest`
- CUDA 13 for container-aware commands unless `--cuda` is supplied

The architecture mapping is `x86_64`/`amd64` to `x86_64` and `aarch64`/`arm64` to `aarch64`.

The project Dockerfile installs the wrapper-pinned Holoscan CLI on top of the SDK development
image. The live V4L2 image adds model-export dependencies; its CMake target downloads the pinned,
SHA-256-verified checkpoint and writes the exported ONNX descriptor, external tensor data, and
licenses into `data/v4l2_depth`, not into an image layer. The loopback image extends the live image
with GStreamer H.264 decoding/conversion, Python GObject bindings, and V4L2 user-space tools. Model
preparation is enabled by both inference modes because it is unnecessary for compile-only
validation and requires network/export dependencies that the validation image intentionally omits.
The output-based custom command avoids re-exporting an existing complete model.

`loopback` is a metadata-defined mode rather than a special case in the repository wrapper. Its
runner owns the feeder and application processes, waits for an explicit feeder-ready file before
opening the capture side, forwards termination signals, detects unexpected feeder exits, and
unconditionally cleans up both child process groups. Its `/dev/video42`, progressive YUYV,
1280x720, and 25 FPS contract preserves the documented 1920x1080 test video's 16:9 aspect ratio.
The feeder explicitly disables borders, so this specific resize neither pads nor distorts the
frame. The contract is fixed so appended application arguments cannot silently diverge from the
producer format.

## Validation

Run the camera-, model-, and display-independent project workflow with:

```bash
./holohub lint
./holohub list
./holohub env-check
./holohub modes v4l2_depth
./holohub build v4l2_depth
./holohub test v4l2_depth
./holohub run v4l2_depth
```

The CTest driver configures a clean build and runs the `v4l2_depth` help and compile-only graph
tests, both loopback script help surfaces, and metadata resolution for all container targets. The
application defaults to its hardware-free `validate` mode; select `live` for a compatible camera or
`loopback` for the fixed-path test video on a host with `/dev/video42`. Both inference modes require
an X11 session to render their output, although the display operator continues draining frames
without rendering when X11 is unavailable.

After completing the host setup documented in [README.md](README.md), run the hardware-in-the-loop
path with:

```bash
./holohub run v4l2_depth loopback
```

The repository's pre-merge job builds and validates `v4l2_depth` on native x86_64 and SBSA Blossom
workers. Live and loopback capture remain hardware-in-the-loop checks. See
[`ci/README.md`](ci/README.md) for the Jenkins and GitLab configuration and pinned SDK baseline.

## Adding another sample

1. Add operator or application source under the appropriate top-level directory.
2. Add a `metadata.json` with a unique project name; application and operator names must not
   collide because HoloHub discovery resolves projects by name and language.
3. Register the project with `add_holohub_application()` or `add_holohub_operator()`.
4. Declare operator dependencies in the application helper call.
5. Add bounded CTest coverage.
6. Verify discovery, build, test, run, and lint through `./holohub`.
