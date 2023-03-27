# VideoMaster appplications

## Requirements
To build this application the VideoMaster SDK is required.
These applications are built with Holoscan SDK 0.4.

### Build Instructions

Make sure CMake (www.cmake.org) is installed on your system (minimum version 3.20)

- [Holoscan Debian Package](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_dev_deb) - Follow the instructions in the link to install the latest version of Holoscan Debian package from NGC.

- Create a build directory:

  ```bash
  mkdir -p <build_dir> && cd <build_dir>
  ```

- Configure with CMake:

Make sure CMake can find your installation of the Holoscan SDK. For example, setting `holoscan_ROOT` to its install directory during configuration:

  ```bash
  cmake -S <source_dir> -B <build_dir> -Dholoscan_ROOT=<holoscan_sdk_install_dir> -DVideoMaster_SDK_DIR=<videomaster_sdk_install_dir>
  ```

- Build:

  ```bash
  cmake --build <build_dir>
  ```

### Run Instructions

In your build directory, run the following command:

    ```bash
    <build_dir>/videomaster_tool_tracking
    ```
    