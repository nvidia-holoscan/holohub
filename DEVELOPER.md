# Developer Guide - Holoscan Camera

This guide covers the C++ SDK 5.x layout, build system, and CI workflow for
developing and distributing this Holoscan Module.

## Module Layout

```text
holoscan-camera/
├── holoscan_camera                 # CLI wrapper
├── Dockerfile                      # Development container image
├── CMakeLists.txt                  # Root CMake orchestration
├── cmake/container.ctest           # Project-local CTest driver for ./holoscan_camera test
├── ci/                             # GitLab/Jenkins pipeline and lint helpers
├── metadata.json                   # Module metadata
├── operators/v4l2_capture_op/      # C++ V4L2 capture operator and its unit tests
├── applications/holoscan_camera_v4l2/cpp/
└── pkg/holoscan-camera/            # Debian package metadata
```

Operator tests live beside the operator they cover, under
`operators/<op>/tests/`, and are declared while `operators/` is processed.

## Holoscan SDK 5.x EA Baseline

The wrapper does not use an NGC `v5.0.0` image. It expects the Holoscan SDK
5.x EA source tree to be built first. The SDK build creates both the install
tree and the local base image used by container builds.

```bash
cd ../holoscan-sdk/public
./run build \
    --build-python false \
    --build-benchmarks false
export HOLOSCAN_SDK_INSTALL_DIR="$(pwd)/$(./run get_install_dir)"
cd -
```

The wrapper resolves the SDK in this order:

1. `HOLOSCAN_SDK_INSTALL_DIR`, which must name an install tree directly.
2. `HOLOSCAN_SDK_ROOT`, either an install tree or a parent containing `install-<arch>`.
3. A sibling checkout at `../holoscan-sdk/install-<arch>` or `../holoscan-sdk/public/install-<arch>`.
4. `/opt/nvidia/holoscan` when already inside an SDK-derived container.

It defaults the base image to `holoscan-sdk-build-x86_64:latest` or
`holoscan-sdk-build-aarch64:latest` and passes `holoscan_ROOT` to CMake.

## Wrapper Commands

| Command | What it does |
| --- | --- |
| `./holoscan_camera run-container` | Build and start the development container |
| `./holoscan_camera build holoscan_camera_v4l2 --language cpp` | Build the C++ validation application |
| `./holoscan_camera run holoscan_camera_v4l2 --language cpp` | Build and run the C++ validation application |
| `./holoscan_camera test holoscan_camera_v4l2 --language cpp` | Run CTest via `cmake/container.ctest` |
| `./holoscan_camera package holoscan-camera --pkg-generator DEB` | Build the Debian package |

The wrapper owns the `holoscan-cli` version through
`HOLOSCAN_CLI_PINNED_VERSION`. Set `HOLOSCAN_CLI_INSTALL_ARGS` to override
package-install arguments or `HOLOSCAN_CLI_SOURCE` to use a local CLI checkout.

## Building Without The Wrapper

```bash
cmake -S . -B build -G Ninja \
    -Dholoscan_ROOT="$HOLOSCAN_SDK_INSTALL_DIR" \
    -DBUILD_ALL=ON \
    -DHOLOSCAN_CAMERA_BUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure -LE gpu
```

The project uses C++20 and requires CMake 4.0 with Holoscan SDK 5.0.0 or newer.

Tests carry the labels `unit` for the operator's gtest suite, `plan` for the
application's `--validate` graph-compilation checks, and `gpu` for the subset of
those that need a visible CUDA device. `-LE gpu` is the exclusion for a host
without one.

## SDK 5.x API Shape

`V4l2CaptureOp` follows the typed operator pattern from the 5.x EA samples:

- `final : public holoscan::Operator<>`
- typed port member `holoscan::Output<holoscan::schema::ImageT> frame`
- `setup(holoscan::OperatorSpec&)` declares the port's tensor contract with
  `produces_tensor`, and registers one lifecycle hook per stage the device owns
- `contract()` returns an explicit `holoscan::Contract`, triggered by the
  reader thread's notification rather than by a clock
- `compute(holoscan::ExecutionContext&)` returns `holoscan::expected<void, holoscan::Error>`

The published payload is a `holoscan::schema::ImageT` carrying a packed YUYV
frame, not a bare tensor, so a consumer reads the geometry, encoding, frame
identifier, and acquisition instant from the sample. Frame placement is a
constructor parameter defaulting to `holoscan::MemoryKind::kHost`; see
[README.md](README.md) for the accepted values and for the explicit device
binding `kCudaDevice` requires.

The application uses `holoscan::Graph`, `graph.op<T>()`, and
`holoscan::compile`, and binds tensor output device placement only when the
requested placement is device-resident.

## Jenkins CI

GitLab-first CI lives in `ci/`. The production Jenkins job should load
`ci/pre-merge-pipeline.groovy` from protected `main`; the script checks out a
synthetic source-into-target merge for merge requests and never loads pipeline
code from an untrusted MR ref.

The GPU flows run against both the pinned SDK revision and the latest `main-5x`
SDK tip on native x86_64 and SBSA workers. The pinned flows are the required
merge baseline; top-of-tree flows are advisory for merge requests.

Each GPU flow:

1. verify native architecture and R580-or-newer driver support;
2. check out the SDK SHA in `ci/holoscan-sdk.version` or the latest `main-5x` tip;
3. build the SDK with Python disabled and benchmarks disabled;
4. resolve `HOLOSCAN_SDK_INSTALL_DIR`;
5. build and test `holoscan_camera_v4l2`; and
6. package `holoscan-camera` as a Debian package.

See [ci/README.md](ci/README.md) for Jenkins credentials, webhook settings,
CDash behavior, and SDK pin updates.

## Packaging

Build a Debian package through the wrapper:

```bash
./holoscan_camera package holoscan-camera --pkg-generator DEB
```

Keep these metadata fields synchronized before publishing:

| Field | Purpose |
| --- | --- |
| `metadata.json:module.version` | Module and Debian package version |
| `metadata.json:module.binary_packages` | Published Debian package name |
| `pkg/holoscan-camera/CMakeLists.txt` | Debian package dependency metadata |

## Naming Conventions

| Context | Convention | Example |
| --- | --- | --- |
| C++ namespace | `snake_case` | `holoscan::holoscan_camera` |
| Repository folder | `holoscan-<slug>` | `holoscan-camera` |
| Debian package | `holoscan-<slug>` | `holoscan-camera` |
| CMake option prefix | `UPPER_SNAKE` | `HOLOSCAN_CAMERA_BUILD_TESTING` |
