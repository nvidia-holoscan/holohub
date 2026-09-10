# Cross Compilation Example

This example cross-compiles a small Holoscan SDK 4.6 application on an x86_64 host for an
AArch64/SBSA target. The builder consumes the released AArch64 Holoscan Debian package; it does not
build the SDK from source or execute AArch64 programs through QEMU.

## Part 1: Build the example

### Requirements

- An x86_64 Linux host with Docker Engine and BuildKit.
- Internet access while building the image.
- `file` and `readelf` (`binutils`) on the host for the final artifact checks.

A GPU and target hardware are not required to cross-compile. The example uses CUDA architecture
`100`, matching an NVIDIA Thor target. Change it to the SM architecture of your target when needed.

### Build the cross-compilation image

From the HoloHub checkout:

```bash
cd applications/cross_compilation_example
docker build --platform linux/amd64 \
  -t holoscan-4-cross-sbsa:4.6.0 .
```

The image installs the native x86_64 build tools and CUDA Cross-SBSA packages, then uses NVIDIA's
signed SBSA apt repository to download and extract `holoscan-cuda-13_4.6.0.0-1_arm64.deb` into
`/opt/nvidia/holoscan`. Extracting the target package makes its headers, CMake exports, and AArch64
libraries available without trying to execute its AArch64 package dependencies in the x86_64
builder.

### Configure, build, and stage

Create an ignored output directory and run each CMake step in the same image. Mounting the source
for all three steps allows CMake to regenerate the Ninja files when necessary.

```bash
mkdir -p build-cross

docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace/source,readonly \
  --mount type=bind,src="$PWD/build-cross",dst=/workspace/build \
  holoscan-4-cross-sbsa:4.6.0 \
  cmake -S /workspace/source -B /workspace/build -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=/opt/holoscan-cross/aarch64-cross-sbsa.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DCMAKE_INSTALL_PREFIX=/workspace/build/install

docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace/source,readonly \
  --mount type=bind,src="$PWD/build-cross",dst=/workspace/build \
  holoscan-4-cross-sbsa:4.6.0 \
  cmake --build /workspace/build

docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,src="$PWD",dst=/workspace/source,readonly \
  --mount type=bind,src="$PWD/build-cross",dst=/workspace/build \
  holoscan-4-cross-sbsa:4.6.0 \
  cmake --install /workspace/build
```

The staged executable is `build-cross/install/bin/cross_compilation_example`.

### Verify the result

```bash
file build-cross/install/bin/cross_compilation_example
readelf --dynamic build-cross/install/bin/cross_compilation_example | grep libholoscan_core
```

`file` must report an AArch64 ELF executable. `readelf` must report a dependency on
`libholoscan_core.so.4`. Do not use host `ldd`: the host cannot load the AArch64 executable.

To run it, copy the executable to a compatible AArch64 target with the matching Holoscan 4.6.0
CUDA 13 runtime installed. This example stages only the application executable; it does not bundle
the SDK or other shared libraries.

## Part 2: Add Holoscan to an existing project

This section starts with a plain Ubuntu 24.04 x86_64 build environment and keeps the target SDK
under the existing project's `.cross/` directory. The build machine does not need an NVIDIA GPU.

### Install the build tools

Confirm that the build machine is x86_64, then install the native tools and GNU AArch64 compiler:

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

Holoscan 4.6 exports require CMake 3.30.4 or newer. Ubuntu 24.04 provides an older version, so
install the pinned version in a project-local Python environment:

```bash
python3 -m venv .cross/cmake-venv
. .cross/cmake-venv/bin/activate
python -m pip install cmake==4.0.3
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
CUDA headers and libraries under `/usr/local/cuda-13.0/targets/sbsa-linux`.

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
sudo apt-get update \
  -o Dir::Etc::sourcelist=sources.list.d/nvidia-holoscan-sbsa.list \
  -o Dir::Etc::sourceparts=- \
  -o APT::Architecture=arm64 \
  -o APT::Architectures=arm64

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
sudo dpkg --add-architecture arm64

sudo apt-get update \
  -o Dir::Etc::sourcelist=sources.list.d/nvidia-holoscan-sbsa.list \
  -o Dir::Etc::sourceparts=-

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

curl -fsSL "${HOLOSCAN_DEB_URL}" -o "${HOLOSCAN_DEB}"
printf '%s  %s\n' "${HOLOSCAN_DEB_SHA256}" "${HOLOSCAN_DEB}" | sha256sum --check -
```

#### Validate and extract the package

After completing one download option, validate and extract the package:

```bash
test -f "${HOLOSCAN_DEB}"
test "$(dpkg-deb -f "${HOLOSCAN_DEB}" Architecture)" = arm64
test "$(dpkg-deb -f "${HOLOSCAN_DEB}" Version)" = "${HOLOSCAN_DEB_VERSION}"
dpkg-deb --extract "${HOLOSCAN_DEB}" .cross/holoscan-root

HOLOSCAN_TARGET_SDK_ROOT="$PWD/.cross/holoscan-root/opt/nvidia/holoscan"
test -f "${HOLOSCAN_TARGET_SDK_ROOT}/include/holoscan/holoscan.hpp"
test -f "${HOLOSCAN_TARGET_SDK_ROOT}/lib/cmake/holoscan/holoscan-config.cmake"
file -L "${HOLOSCAN_TARGET_SDK_ROOT}/lib/libholoscan_core.so"
```

The final command must report an ARM AArch64 library. The apt options verify the repository
signature and package hash; the curl option relies on the explicit SHA-256 check. Add `.cross/` to
the project's `.gitignore`. The target needs the matching runtime package from the
[Holoscan SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation).

### Add the CMake toolchain

Copy [`aarch64-cross-sbsa.cmake`](cmake/aarch64-cross-sbsa.cmake) and
[`nvcc-cross-sbsa`](cmake/nvcc-cross-sbsa) into the existing project's `cmake/` directory, then
make the wrapper executable:

```bash
chmod +x cmake/nvcc-cross-sbsa
```

The toolchain accepts these project-specific inputs and sets the remaining CMake cross-compilation
variables:

| Variable | Value or role |
| --- | --- |
| `CMAKE_SYSTEM_NAME`, `CMAKE_SYSTEM_PROCESSOR` | Select Linux and AArch64 as the target |
| `HOLOSCAN_CROSS_GNU_PREFIX` | Prefix for the `aarch64-linux-gnu` cross tools |
| `HOLOSCAN_CROSS_SYSROOT` | GNU cross-tool root, `/usr/aarch64-linux-gnu` by default |
| `HOLOSCAN_CROSS_CUDA_ROOT` | Native CUDA toolkit with Cross-SBSA support, `/usr/local/cuda-13.0` by default |
| `HOLOSCAN_TARGET_SDK_ROOT` | Extracted AArch64 SDK prefix, `/opt/nvidia/holoscan` by default |
| `CMAKE_CUDA_COMPILER` | The included wrapper that always selects CUDA's `sbsa-linux` target |
| `CMAKE_CUDA_HOST_COMPILER` | `aarch64-linux-gnu-g++` |
| `CUDAToolkit_ROOT`, `CUDAToolkit_NVCC_EXECUTABLE` | Keep CUDA discovery on the native compiler and SBSA target toolkit |
| `CMAKE_FIND_ROOT_PATH` | Restricts target lookup to Holoscan, GNU AArch64, and CUDA SBSA roots |
| `CMAKE_FIND_ROOT_PATH_MODE_*` | Finds host programs but target packages, headers, and libraries |

`CMAKE_SYSROOT` is deliberately not set. Ubuntu's GNU cross compiler already has its target
sysroot, while Holoscan and CUDA are separate prefixes. Forcing one global sysroot would incorrectly
rewrite those absolute package paths.

Set `CMAKE_CUDA_ARCHITECTURES` explicitly when the project enables CUDA. It is not hard-coded in
the toolchain because it is a property of the deployment GPU, not the compiler container.

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
cmake ... -DCMAKE_CUDA_ARCHITECTURES=100
```

Link additional exported targets only when the code uses them. For example, an application using
the packaged ping operators would add `holoscan::ops::ping_tx` and `holoscan::ops::ping_rx` to
`target_link_libraries`.

### Configure and build

Activate the CMake environment, point the toolchain at the project-local SDK, and build:

```bash
. .cross/cmake-venv/bin/activate

cmake -S . -B build-cross -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/aarch64-cross-sbsa.cmake" \
  -DHOLOSCAN_TARGET_SDK_ROOT="$PWD/.cross/holoscan-root/opt/nvidia/holoscan" \
  -DHOLOSCAN_CROSS_CUDA_ROOT=/usr/local/cuda-13.0 \
  -DHOLOSCAN_CROSS_GNU_PREFIX=aarch64-linux-gnu- \
  -DHOLOSCAN_CROSS_SYSROOT=/usr/aarch64-linux-gnu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build-cross/install"

cmake --build build-cross
cmake --install build-cross
```

For a CUDA project targeting Thor, also configure with `-DCMAKE_CUDA_ARCHITECTURES=100`. Use the
SM architecture of the actual deployment GPU for other targets.

Every additional native library used by the project also needs AArch64 headers and libraries.
Check CMake's configure output to ensure it never satisfies a target dependency with an x86_64
library from the build machine.

### Deployment boundary

Cross-compilation proves that the application configures and links for AArch64. It does not prove
runtime behavior. The target must provide a compatible Linux ABI, CUDA runtime/driver, matching
Holoscan 4.6 libraries, and every application-specific shared library. Jetson multimedia or other
BSP-specific dependencies may require a sysroot from the exact target OS rather than the generic
SBSA roots used here.
