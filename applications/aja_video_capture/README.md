# AJA Capture

Minimal example to demonstrate the use of the aja source operator to capture device input and stream to holoviz operator.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/aja_setup.html) to setup the AJA Card.*

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/aja_capture/cpp/aja_capture
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/aja_capture/cpp/aja_capture
  
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/aja_capture/cpp/aja_capture
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/aja_capture.py
  ```
* **using deb package install**:
  ```bash
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/aja_capture/python/aja_capture.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/aja_capture/python/aja_capture.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/aja_capture/python/aja_capture.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/aja_capture/python/aja_capture.py
  ```
## Settings

 To evaluate the AJA example using alternative resolutions, you may modify the aja_capture.yaml configuration file as needed. For instance, to test a resolution format of 1280 x 720 at 60 Hz, you can specify the following parameters in the aja section of the configuration :
   
    ```bash
      aja:
        width: 1280
        height: 720
        framerate: 60
    ```