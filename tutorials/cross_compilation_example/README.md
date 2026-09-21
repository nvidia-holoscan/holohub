# Cross Compilation Example

This tutorial shows how to cross-compile a small Holoscan Software Development Kit (SDK) 4.6
application on an x86_64 host machine for NVIDIA Jetson AGX Thor and NVIDIA IGX Thor target
systems. Follow along to cross-compile with the Holoscan SDK arm64 Debian package without rebuilding
the SDK from source or emulating the target environment with QEMU.

## Background

This tutorial is intended for Jetson and IGX embedded developers who want to build Holoscan C++
applications on an x86_64 Linux workstation or Continuous Integration (CI) system and deploy them
to an AArch64 target system. Cross-compilation runs the compiler on one architecture while producing
binaries for another.

The pinned workflow targets CUDA 13 Server Base System Architecture (SBSA) systems using NVIDIA's
generic AArch64 packages. Jetson Linux 38 aligned Jetson AGX Thor with SBSA, and NVIDIA's CUDA
Cross-SBSA packages support cross-platform development for arm64 Jetson Thor and SBSA targets. The
same build is therefore suitable for the core Holoscan API on these target baselines:

| Target | Supported software baseline |
| --- | --- |
| NVIDIA Jetson AGX Thor Developer Kit | JetPack 7.0, Jetson Linux 38.2, Ubuntu 24.04, and CUDA 13 |
| NVIDIA IGX Thor Developer Kit and Developer Kit Mini | IGX Software (IGX-SW) 2.0 Production Release, Board Support Package (BSP) 38.5.0, Ubuntu 24.04, and CUDA 13 |

See the [Jetson Linux 38.2 release notes](https://docs.nvidia.com/jetson/archives/r38.2/ReleaseNotes/Jetson_Linux_Release_Notes_r38.2.pdf),
[IGX-SW 2.0 release notes](https://docs.nvidia.com/igx/user-guide/2.0/software-releases/software-release-2-0-thor-notes-pr.html),
and [CUDA cross-platform installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#cuda-cross-platform-installation)
for the platform details.

SBSA standardizes the AArch64 platform interfaces used by this example, so the core Holoscan
application can use NVIDIA's published generic CUDA and Holoscan packages without a board-specific
target filesystem. The example intentionally uses Ubuntu 24.04's AArch64 GNU Compiler Collection
(GCC) cross-compiler and target libraries, matching the Ubuntu 24.04 Holoscan package. It does not use
the standalone Jetson Linux 38.2 cross-toolchain because that toolchain includes a GNU C Library
(glibc) 2.28 sysroot, which is older than the glibc required by the released Ubuntu 24.04 Holoscan
package.

This supported path is limited to the core Holoscan C++ API demonstrated here. Applications that use
board-specific camera, multimedia, networking, or other BSP libraries need matching target packages
and a sysroot captured from the target OS. Those dependencies are outside this tutorial's scope.

A sysroot is a directory tree that mirrors the target system's filesystem and supplies the target
headers and libraries used during cross-compilation instead of the host's filesystem.

This separates the build environment from the deployment environment. The x86_64 builder contains
the compiler, CMake, headers, and CUDA cross-compilation packages, while the embedded target needs
only the application and its matching runtime dependencies. Developers can use workstation or CI
resources, create repeatable container builds, avoid installing a complete build toolchain on the
target, and build without keeping the target connected. The resulting application must still be
tested on the target hardware.

One alternative to cross-compilation is to fully emulate the target build environment. QEMU is an
emulator and virtualizer that can run AArch64 programs on an x86_64 host by translating their
instructions. It is useful when a build needs to execute target binaries, but emulation adds runtime
overhead and configuration. This tutorial instead runs the build tools natively on x86_64, uses a
cross-compiler to emit AArch64 code, and extracts the target Holoscan package without running its
programs. Avoiding emulation makes the build faster and simpler, but the produced executable cannot
be run on the build host.

## What you will learn

This tutorial covers two workflows:

1. **Build the containerized example:** Create the cross-compilation container with Docker, build
   and stage the example, verify that the result is an AArch64 executable linked to Holoscan, and
   validate it on a supported Thor target.
2. **Add Holoscan to an existing project:** Install the host and CUDA cross-compilation tools,
   acquire and extract the AArch64 Holoscan SDK package, configure the CMake toolchain, discover and
   link Holoscan targets, build the project, and understand the deployment boundary.

## Part 1: Build the example

### Requirements

- An x86_64 Linux host with Docker Engine and BuildKit.
- Internet access while building the image.
- Python 3 with virtual-environment support for the direct Holoscan CLI workflow.
- `file` and `readelf` (`binutils`) on the host for the final artifact checks.
- Optional: A Jetson AGX Thor Developer Kit with JetPack 7.0, or an IGX Thor Developer Kit or
  Developer Kit Mini with IGX-SW 2.0, for deployment and runtime validation.

A graphics processing unit (GPU) and target hardware are not required to cross-compile. This C++
example does not compile CUDA source, so it does not set `CMAKE_CUDA_ARCHITECTURES`. For projects
that compile CUDA source, this tutorial uses `75-virtual` as a portable starting point. It embeds
NVIDIA Parallel Thread Execution (PTX) code that a compatible CUDA driver compiles for the target
GPU using just-in-time (JIT) compilation. When the deployment GPU is known, replace it with the
appropriate hardware-specific architecture, as described in
[Choose a CUDA target architecture](#choose-a-cuda-target-architecture).

### Using the Holoscan CLI directly

Create a Python virtual environment and install the Holoscan CLI version pinned by this tutorial:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --extra-index-url https://pypi.nvidia.com \
  "holoscan-cli==4.6.0"
```

From the repository root, build and install the example with one command:

```bash
HOLOSCAN_CLI_BASE_SDK_VERSION=4.6.0 \
  holoscan install cross_compilation_example
```

`HOLOSCAN_CLI_BASE_SDK_VERSION` supplies the SDK image version that the standalone CLI does not
assume.

### Using the HoloHub CLI

As an alternative, run the HoloHub wrapper from the repository root:

```bash
./holohub install cross_compilation_example
```

The wrapper manages the compatible Holoscan CLI environment and SDK version, so this option does
not require a separate Python virtual environment or `HOLOSCAN_CLI_BASE_SDK_VERSION`.

The default `cross_compile` mode builds the x86_64 cross-compilation container, configures the
project with `aarch64-cross-sbsa.cmake`, compiles an AArch64 executable, and installs it to
`install/bin/cross_compilation_example`. The mode uses Docker's standard `runc` runtime because a
GPU is not required for this build.

Continue with [Verify the result](#verify-the-result), using the CLI artifact path:

```bash
CROSS_COMPILED_APP=install/bin/cross_compilation_example
```

### Manual Docker and CMake workflow

The following steps show each Docker and CMake command performed by the CLI workflow.

#### Build the cross-compilation image

From the repository root:

```bash
docker build --platform linux/amd64 \
  -f tutorials/cross_compilation_example/Dockerfile \
  -t holoscan-4-cross-sbsa:4.6.0 .
```

The image installs the native x86_64 build tools, Ubuntu 24.04's AArch64 compiler and target
libraries, and CUDA Cross-SBSA packages. It then uses NVIDIA's signed SBSA apt repository to download
and extract `holoscan-cuda-13_4.6.0.0-1_arm64.deb` into `/opt/nvidia/holoscan`. Extracting the target
package makes its headers, CMake exports, and AArch64 libraries available without trying to execute
its AArch64 package dependencies in the x86_64 builder.

#### Configure, build, and stage

Create an ignored output directory and run each CMake step in the same image. Mounting the source
for all three steps allows CMake to regenerate the Ninja files when necessary.

```bash
cd tutorials/cross_compilation_example
mkdir -p build-cross

# Configure
docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace/source,readonly \
  --mount type=bind,src="$PWD/build-cross",dst=/workspace/build \
  holoscan-4-cross-sbsa:4.6.0 \
  cmake -S /workspace/source -B /workspace/build -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=/opt/holoscan-cross/aarch64-cross-sbsa.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/workspace/build/install

# Build
docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace/source,readonly \
  --mount type=bind,src="$PWD/build-cross",dst=/workspace/build \
  holoscan-4-cross-sbsa:4.6.0 \
  cmake --build /workspace/build

# Install
docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace/source,readonly \
  --mount type=bind,src="$PWD/build-cross",dst=/workspace/build \
  holoscan-4-cross-sbsa:4.6.0 \
  cmake --install /workspace/build
```

The staged executable is `build-cross/install/bin/cross_compilation_example`. Set the artifact path
for the remaining steps:

```bash
CROSS_COMPILED_APP=build-cross/install/bin/cross_compilation_example
```

### Verify the result

```bash
file "${CROSS_COMPILED_APP}"
readelf --dynamic "${CROSS_COMPILED_APP}" | grep libholoscan_core
```

`file` must report an AArch64 Executable and Linkable Format (ELF) binary. `readelf` must report a
dependency on `libholoscan_core.so.4`. Do not use host `ldd`: the host cannot load the AArch64
executable.

### Deploy and validate on a Thor target

Prepare either supported target baseline listed in [Background](#background), then install the
Holoscan 4.6 CUDA 13 Debian package by following the
[Holoscan SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation).
If the NVIDIA SBSA repository is already configured on the target, the package used by this build
can be installed directly:

```bash
sudo apt-get update
sudo apt-get install --yes holoscan-cuda-13=4.6.0.0-1
```

On the target, confirm the platform and runtime versions:

```bash
test "$(uname -m)" = aarch64
. /etc/os-release
test "${VERSION_ID}" = 24.04
dpkg-query -W -f='${Package} ${Version}\n' holoscan-cuda-13
dpkg-query -W -f='${Package} ${Version}\n' cuda-cudart-13-0
head -n 1 /etc/nv_tegra_release
```

The Holoscan package must report `4.6.0.0-1`, the CUDA runtime must report a 13.0 release, and the
Jetson or IGX release information must match the selected baseline. Copy the staged executable from
the x86_64 host, replacing the target login and address:

```bash
scp "${CROSS_COMPILED_APP}" \
  <user>@<target>:/tmp/cross_compilation_example
```

Run the remaining checks on the target:

```bash
chmod +x /tmp/cross_compilation_example

# Validate that all shared libraries resolve on the target
ldd /tmp/cross_compilation_example | tee /tmp/cross_compilation_example.ldd
! grep -q 'not found' /tmp/cross_compilation_example.ldd

# Validate the application runs successfully
/tmp/cross_compilation_example 2>&1 | tee /tmp/cross_compilation_example.log
grep -F 'Hello Holoscan!' /tmp/cross_compilation_example.log
```

The host-side `file` and `readelf` checks validate the cross-build and linkage. This target-side run
validates the runtime ABI, Holoscan and CUDA dependencies, and application behavior on Jetson AGX
Thor or IGX Thor. This example stages only the application executable; it does not bundle the SDK or
other shared libraries.

## Part 2: Add Holoscan to an existing project

This section starts with a plain Ubuntu 24.04 x86_64 build environment and keeps the target SDK
under the existing project's `.cross/` directory. The build machine does not need an NVIDIA GPU.

### Install the build tools

Confirm that the build machine is x86_64, then install the native tools and Ubuntu 24.04 GNU
Compiler Collection (GCC) AArch64 cross-compiler and target libraries:

```bash
test "$(dpkg --print-architecture)" = amd64

sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
  binutils-aarch64-linux-gnu \
  ca-certificates \
  curl \
  file \
  g++-aarch64-linux-gnu \
  gnupg \
  ninja-build \
  python3-venv
```

The Holoscan 4.6 package declares CMake 3.20 as its minimum. This cross-compilation workflow is
tested with CMake 3.30.4 or newer, so install a compatible version in a project-local Python
environment:

```bash
python3 -m venv .cross/cmake-venv
. .cross/cmake-venv/bin/activate
python -m pip install 'cmake>=3.30.4'
cmake --version
```

### Install the CUDA cross-compilation packages

Add the NVIDIA x86_64 and Cross-SBSA repositories:

```bash
CUDA_KEYRING=/usr/share/keyrings/cuda-archive-keyring.gpg

curl -fsSL \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
  | sudo gpg --dearmor --yes --output "${CUDA_KEYRING}"

echo "deb [signed-by=${CUDA_KEYRING}] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/nvidia-cuda-x86_64.list >/dev/null
echo "deb [signed-by=${CUDA_KEYRING}] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/cross-linux-sbsa/ /" \
  | sudo tee /etc/apt/sources.list.d/nvidia-cuda-cross-sbsa.list >/dev/null
echo "deb [arch=arm64 signed-by=${CUDA_KEYRING}] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/ /" \
  | sudo tee /etc/apt/sources.list.d/nvidia-holoscan-sbsa.list >/dev/null

sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
  cuda-cross-sbsa-13-0 \
  cuda-nvcc-13-0
```

`cuda-nvcc-13-0` supplies the native x86_64 compiler. `cuda-cross-sbsa-13-0` supplies the AArch64
CUDA headers and libraries under `/usr/local/cuda-13.0/targets/sbsa-linux`. NVIDIA documents this
Cross-SBSA package as the supported CUDA cross-platform toolkit for arm64 Jetson Thor and SBSA
targets.

### Extract the AArch64 Holoscan package

Choose one of the following methods to download the released target package. Each method writes the
same file, which is validated and extracted afterward:

```bash
HOLOSCAN_DEB_VERSION=4.6.0.0-1
HOLOSCAN_DEB_NAME="holoscan-cuda-13_${HOLOSCAN_DEB_VERSION}_arm64.deb"
HOLOSCAN_DEB="$PWD/.cross/${HOLOSCAN_DEB_NAME}"
mkdir -p .cross/holoscan-root
```

#### Option 1: Use temporary apt architecture overrides

This is the least invasive option. The architecture settings apply only to these commands and do
not register arm64 as a foreign architecture for the build machine:

```bash
# Update the target repository metadata
sudo apt-get update \
  -o Dir::Etc::sourcelist=sources.list.d/nvidia-holoscan-sbsa.list \
  -o Dir::Etc::sourceparts=- \
  -o APT::Architecture=arm64 \
  -o APT::Architectures=arm64

# Download the package
(
  cd .cross
  apt-get download \
    -o APT::Architecture=arm64 \
    -o APT::Architectures=arm64 \
    "holoscan-cuda-13:arm64=${HOLOSCAN_DEB_VERSION}"
)
```

#### Option 2: Register arm64 with dpkg

Use this option when standard Debian multiarch configuration is preferred. Registering the
architecture is system-wide, but the isolated update still prevents apt from querying arm64
packages from every configured Ubuntu repository:

```bash
# Register arm64 as a foreign architecture
sudo dpkg --add-architecture arm64

# Update the target repository metadata
sudo apt-get update \
  -o Dir::Etc::sourcelist=sources.list.d/nvidia-holoscan-sbsa.list \
  -o Dir::Etc::sourceparts=-

# Download the package
(
  cd .cross
  apt-get download "holoscan-cuda-13:arm64=${HOLOSCAN_DEB_VERSION}"
)

# Optional if arm64 was not already configured:
sudo dpkg --remove-architecture arm64
```

The removal command will fail if arm64 packages are installed. Omit it if arm64 was already
registered or other workflows need it.

#### Option 3: Download the package directly with curl

This avoids apt architecture configuration, but the package URL and checksum must be updated
together whenever the version changes:

```bash
HOLOSCAN_DEB_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/${HOLOSCAN_DEB_NAME}"
HOLOSCAN_DEB_SHA256=de5da74f095c36f64ba446a04f98e8f3e30c04be96845b604333098f95e6e5e9

# Download the package
curl -fsSL "${HOLOSCAN_DEB_URL}" -o "${HOLOSCAN_DEB}"

# Validate the checksum
printf '%s  %s\n' "${HOLOSCAN_DEB_SHA256}" "${HOLOSCAN_DEB}" | sha256sum --check -
```

#### Validate and extract the package

After completing one download option, validate and extract the package:

```bash
# Validate the package
test -f "${HOLOSCAN_DEB}"
test "$(dpkg-deb -f "${HOLOSCAN_DEB}" Architecture)" = arm64
test "$(dpkg-deb -f "${HOLOSCAN_DEB}" Version)" = "${HOLOSCAN_DEB_VERSION}"

# Extract to disk
dpkg-deb --extract "${HOLOSCAN_DEB}" .cross/holoscan-root

# Validate the extraction succeeded
HOLOSCAN_TARGET_SDK_ROOT="$PWD/.cross/holoscan-root/opt/nvidia/holoscan"
test -f "${HOLOSCAN_TARGET_SDK_ROOT}/include/holoscan/holoscan.hpp"
test -f "${HOLOSCAN_TARGET_SDK_ROOT}/lib/cmake/holoscan/holoscan-config.cmake"
file -L "${HOLOSCAN_TARGET_SDK_ROOT}/lib/libholoscan_core.so"
```

The final command must report an ARM AArch64 library. The apt options verify the repository
signature and package hash; the curl option relies on the explicit SHA-256 check. Add `.cross/` to
the project's `.gitignore`. The target needs the matching runtime package from the
[Holoscan SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation).
Using the same `holoscan-cuda-13` version in the extracted build dependency and on the target avoids
an SDK application binary interface (ABI) mismatch.

Extracting this package is sufficient for the core `holoscan::core` API used by this tutorial. It
is not equivalent to installing every optional Holoscan backend and its development dependencies.
Installing `holoscan-cuda-13:arm64` into the x86_64 builder would mix target packages into the host
package database and could run AArch64 package-maintainer scripts. Keep downloading and extracting
target packages for the cross-build instead. On the AArch64 target system, use `apt-get install` so
apt can install and configure the matching runtime dependency closure normally.

### Add the CMake toolchain

Copy [`aarch64-cross-sbsa.cmake`](cmake/aarch64-cross-sbsa.cmake) and
[`nvcc-cross-sbsa`](cmake/nvcc-cross-sbsa) into the existing project's `cmake/` directory.

The toolchain accepts the project-specific roots below and sets the remaining CMake
cross-compilation variables. It uses the standard compiler and CUDA locations installed above:

| Variable | Value or role |
| --- | --- |
| `CMAKE_SYSTEM_NAME`, `CMAKE_SYSTEM_PROCESSOR` | Select Linux and AArch64 as the target |
| `HOLOSCAN_CROSS_SYSROOT` | GNU cross-tool root, `/usr/aarch64-linux-gnu` by default |
| `HOLOSCAN_TARGET_SDK_ROOT` | Extracted AArch64 SDK prefix, `/opt/nvidia/holoscan` by default |
| `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` | Standard `aarch64-linux-gnu` cross compilers |
| `CMAKE_CUDA_COMPILER` | The included wrapper around `/usr/local/cuda-13.0/bin/nvcc` that selects CUDA's `sbsa-linux` target |
| `CMAKE_CUDA_HOST_COMPILER` | `aarch64-linux-gnu-g++` |
| `CUDAToolkit_ROOT`, `CUDAToolkit_NVCC_EXECUTABLE` | Keep CUDA discovery on the native NVIDIA CUDA Compiler (NVCC) and SBSA target toolkit |
| `CMAKE_FIND_ROOT_PATH` | Restricts target lookup to Holoscan, GNU AArch64, and CUDA SBSA roots |
| `CMAKE_FIND_ROOT_PATH_MODE_*` | Finds host programs but target packages, headers, and libraries |

`CMAKE_SYSROOT` is deliberately not set. Ubuntu's GNU cross compiler already has its target
sysroot, while Holoscan and CUDA are separate prefixes. Forcing one global sysroot would incorrectly
rewrite those absolute package paths.

### Choose a CUDA target architecture

Set `CMAKE_CUDA_ARCHITECTURES` only when the project enables CUDA. It is not hard-coded in the
toolchain because it is a property of the deployment GPU, not the compiler container.

In `75-virtual`, `75` means CUDA compute capability 7.5 and `virtual` tells CMake to embed PTX
instead of machine code for a particular GPU. At runtime, the CUDA driver JIT-compiles that PTX for
the installed GPU. This is more portable across compatible GPUs, at the cost of JIT work when the
application first loads its CUDA code.

For a hardware-specific build:

1. Find the deployment GPU's compute capability in NVIDIA's
   [compute capability documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
   When `nvidia-smi` is available on the target, query it directly:

   ```bash
   nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
   ```

2. Remove the decimal point and append `-real`. For example, compute capability 8.7 becomes
   `-DCMAKE_CUDA_ARCHITECTURES="87-real"`.

The `-real` suffix generates GPU machine code and avoids PTX JIT for that architecture, but the
result is less portable. Omitting the suffix, such as `87`, makes CMake generate both real and
virtual code. See CMake's
[`CUDA_ARCHITECTURES` documentation](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)
for the complete syntax.

### Find and link Holoscan

A C++-only project needs this minimum:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_holoscan_app LANGUAGES CXX)

find_package(holoscan 4.6 REQUIRED CONFIG)

add_executable(my_holoscan_app main.cpp)
target_link_libraries(my_holoscan_app PRIVATE holoscan::core)

install(TARGETS my_holoscan_app RUNTIME DESTINATION bin)
```

If the project contains `.cu` files, enable CUDA and pass the target architecture while configuring:

```cmake
project(my_holoscan_app LANGUAGES CXX CUDA)
```

```bash
cmake ... -DCMAKE_CUDA_ARCHITECTURES="75-virtual"
```

Link additional exported targets only when the code uses them. For example, an application using
the packaged ping operators would add `holoscan::ops::ping_tx` and `holoscan::ops::ping_rx` to
`target_link_libraries`.

### Configure and build

Activate the CMake environment, point the toolchain at the project-local SDK, and build:

```bash
# Activate the project-local CMake environment
. .cross/cmake-venv/bin/activate

# Configure
cmake -S . -B build-cross -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/aarch64-cross-sbsa.cmake" \
  -DHOLOSCAN_TARGET_SDK_ROOT="$PWD/.cross/holoscan-root/opt/nvidia/holoscan" \
  -DHOLOSCAN_CROSS_SYSROOT=/usr/aarch64-linux-gnu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build-cross/install"

# Build
cmake --build build-cross

# Install
cmake --install build-cross
```

For a CUDA project, also configure with the CUDA architecture selected in
[Choose a CUDA target architecture](#choose-a-cuda-target-architecture).

Every additional native library used by the project also needs AArch64 headers and libraries.
Check CMake's configure output to ensure it never satisfies a target dependency with an x86_64
library from the build machine.

### Extend the target sysroot for optional SDK components

The hello-world application does not exercise Holoscan inference backends such as TensorRT,
PyTorch, or Open Neural Network Exchange (ONNX) Runtime. Applications that use these components
must add their AArch64 development headers and libraries to the cross-build and install their
matching runtime libraries on the target.

Keep these additions in a separate target package root rather than installing them into the x86_64
builder. A general workflow is:

1. Declare the required AArch64 packages and exact versions in a project-owned manifest.
2. Give apt isolated package status, repository metadata, and package-cache directories for arm64.
3. Use `apt-get --download-only` so apt resolves the dependency closure without installing it.
4. Extract every downloaded Debian package into one target package root with `dpkg-deb --extract`.
5. Add that root to `CMAKE_FIND_ROOT_PATH`, then provide any package-specific CMake configuration.
6. Check every linked library's architecture and validate the result on the target system.

The isolated apt directories prevent the target dependency resolver from using the builder's amd64
package state. The source list must describe the same Ubuntu and NVIDIA repositories used by the
target. For the supported Ubuntu 24.04 SBSA baseline, the setup has this form:

```bash
TARGET_APT_ROOT="$PWD/.cross/apt-arm64"
TARGET_PACKAGE_ROOT="$PWD/.cross/target-root"
CUDA_KEYRING=/usr/share/keyrings/cuda-archive-keyring.gpg

mkdir -p \
  "${TARGET_APT_ROOT}/lists/partial" \
  "${TARGET_APT_ROOT}/archives/partial" \
  "${TARGET_PACKAGE_ROOT}"
: > "${TARGET_APT_ROOT}/status"

cat > "${TARGET_APT_ROOT}/sources.list" <<EOF
deb [arch=arm64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://ports.ubuntu.com/ubuntu-ports noble main restricted universe multiverse
deb [arch=arm64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://ports.ubuntu.com/ubuntu-ports noble-updates main restricted universe multiverse
deb [arch=arm64 signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://ports.ubuntu.com/ubuntu-ports noble-security main restricted universe multiverse
deb [arch=arm64 signed-by=${CUDA_KEYRING}] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/ /
EOF

APT_TARGET_OPTIONS=(
  -o APT::Architecture=arm64
  -o APT::Architectures=arm64
  -o "Dir::Etc::sourcelist=${TARGET_APT_ROOT}/sources.list"
  -o Dir::Etc::sourceparts=-
  -o "Dir::State::status=${TARGET_APT_ROOT}/status"
  -o "Dir::State::lists=${TARGET_APT_ROOT}/lists"
  -o "Dir::Cache::archives=${TARGET_APT_ROOT}/archives"
  -o Debug::NoLocking=1
)
```

Define `TARGET_PACKAGES` before running the resolver, as shown in the TensorRT example below. Keep
every entry version-pinned, and review the simulated package transaction before downloading.
Packages that rely on maintainer scripts, alternatives, or triggers to construct their installed
layout might not work through extraction alone. For those packages, reproduce the required links
explicitly or use a sysroot captured from the exact target root filesystem.

#### Add TensorRT development files

The Holoscan package recommends a TensorRT runtime, but its package metadata does not require the
TensorRT development headers. An application that links the Holoscan inference operator needs the
matching AArch64 development packages. For the CUDA 13.0 workflow pinned by this tutorial, extend
the package manifest with an explicitly compatible TensorRT release. See NVIDIA's
[TensorRT Debian installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-debian.html)
for the complete package set:

```bash
TENSORRT_DEB_VERSION=10.13.3.9-1+cuda13.0

TARGET_PACKAGES=(
  "libnvinfer-dev:arm64=${TENSORRT_DEB_VERSION}"
  "libnvinfer-plugin-dev:arm64=${TENSORRT_DEB_VERSION}"
  "libnvonnxparsers-dev:arm64=${TENSORRT_DEB_VERSION}"
)

# Review the dependency closure without changing the builder
apt-get "${APT_TARGET_OPTIONS[@]}" update
apt-get "${APT_TARGET_OPTIONS[@]}" \
  --simulate --no-install-recommends install \
  "${TARGET_PACKAGES[@]}"

# Download the reviewed dependency closure
apt-get "${APT_TARGET_OPTIONS[@]}" \
  --download-only --no-install-recommends --yes install \
  "${TARGET_PACKAGES[@]}"

# Extract the packages without running AArch64 maintainer scripts
while IFS= read -r -d '' package; do
  dpkg-deb --extract "${package}" "${TARGET_PACKAGE_ROOT}"
done < <(find "${TARGET_APT_ROOT}/archives" -maxdepth 1 -name '*.deb' -print0)
```

The previously extracted Holoscan package remains in `.cross/holoscan-root`; do not add it to this
manifest. Resolving its package-level dependencies would download an AArch64 CUDA compiler even
though the cross-build intentionally uses native x86_64 NVCC. Use the same TensorRT release on the
target. Do not accept the repository's unpinned latest version: the SBSA repository can contain
TensorRT builds for newer CUDA releases.

TensorRT's Debian packages provide the headers and libraries, but a consuming project must also
provide CMake package discovery that defines the imported targets expected by Holoscan. If the
project has a `FindTensorRT.cmake` module that defines `TensorRT::nvinfer_plugin` and
`TensorRT::nvonnxparser`, configure it before Holoscan and link the inference operator:

```cmake
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED MODULE)
find_package(holoscan 4.6 REQUIRED CONFIG)

target_link_libraries(my_holoscan_app PRIVATE holoscan::ops::inference)
```

#### Add PyTorch development files

PyTorch is not supplied by the generic SBSA apt package set used above. Obtain an AArch64 PyTorch
wheel that matches the target's JetPack or IGX-SW release, CUDA version, Python application binary
interface, and the version supported by the selected Holoscan release. Check the
[Holoscan SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation)
for the tested LibTorch version. Do not install or import the AArch64 wheel on the x86_64 builder. A
wheel is a ZIP archive, so extract it into a separate target prefix:

```bash
PYTORCH_WHEEL=/path/to/the/matching-aarch64-pytorch.whl
PYTORCH_TARGET_ROOT="$PWD/.cross/pytorch-root"

python3 -m zipfile -e "${PYTORCH_WHEEL}" "${PYTORCH_TARGET_ROOT}"
test -f "${PYTORCH_TARGET_ROOT}/torch/share/cmake/Torch/TorchConfig.cmake"
file -L "${PYTORCH_TARGET_ROOT}/torch/lib/libtorch.so"
```

The final command must report an ARM AArch64 library. Point CMake at the extracted configuration,
discover Torch before Holoscan, and explicitly link the Holoscan Torch backend:

```cmake
find_package(Torch REQUIRED CONFIG)
find_package(holoscan 4.6 REQUIRED CONFIG)

target_link_libraries(my_holoscan_app PRIVATE
  holoscan::ops::inference
  holoscan::infer::torch)
```

Pass the path while configuring because the shell variable is not automatically visible in CMake:

```bash
cmake ... \
  -DTorch_DIR="$PWD/.cross/pytorch-root/torch/share/cmake/Torch"
```

ONNX Runtime follows the same pattern: provide a compatible AArch64 SDK, make its CMake package
define `ONNXRuntime::ONNXRuntime`, and link `holoscan::infer::onnx_runtime` when that backend is used.

#### Add the package root to CMake and validate it

In the project's copy of `aarch64-cross-sbsa.cmake`, add the optional package root before setting
the `CMAKE_FIND_ROOT_PATH_MODE_*` variables:

```cmake
set(HOLOSCAN_TARGET_PACKAGE_ROOT "" CACHE PATH
  "Root containing additional extracted AArch64 packages")
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  HOLOSCAN_TARGET_PACKAGE_ROOT)

if(HOLOSCAN_TARGET_PACKAGE_ROOT)
  list(PREPEND CMAKE_FIND_ROOT_PATH "${HOLOSCAN_TARGET_PACKAGE_ROOT}")
endif()
```

Pass the package root when configuring:

```bash
cmake ... \
  -DHOLOSCAN_TARGET_PACKAGE_ROOT="$PWD/.cross/target-root"
```

Build an application that actually links the selected backend; the hello-world executable cannot
validate these dependencies. Check representative inputs and the completed executable without
using host `ldd`:

```bash
file -L .cross/target-root/usr/lib/aarch64-linux-gnu/libnvinfer.so
# Run this check only when the PyTorch target prefix was created.
file -L .cross/pytorch-root/torch/lib/libtorch.so
file build-cross/install/bin/my_holoscan_app
readelf --dynamic build-cross/install/bin/my_holoscan_app \
  | grep -E 'libholoscan_infer|libnvinfer|libtorch|libonnxruntime'

if find .cross/target-root -type f \
    \( -name '*.so' -o -name '*.so.*' \) -print0 \
    | xargs -0 file | grep -q 'ELF .*x86-64'; then
  echo 'Found an x86_64 library in an AArch64 target root.' >&2
  exit 1
fi
```

Run the resulting application on the supported Thor target to validate the target driver, ABI,
backend plugins, and runtime package versions. Architecture and link checks on the builder do not
replace that target-side test.

### Deployment boundary

Cross-compilation proves that the application configures and links for AArch64. It does not prove
runtime behavior. The target must provide a compatible Linux application binary interface (ABI),
CUDA runtime/driver, matching Holoscan 4.6 libraries, and every application-specific shared library.
Complete [Deploy and validate on a Thor target](#deploy-and-validate-on-a-thor-target) to validate
the executable on Jetson AGX Thor or IGX Thor hardware. Jetson or IGX camera, multimedia, networking,
and other BSP-specific dependencies require matching packages and may require a sysroot from the
exact target OS rather than the generic SBSA roots used here.
