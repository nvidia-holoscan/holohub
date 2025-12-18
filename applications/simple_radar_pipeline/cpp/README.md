# Simple Radar Pipeline

This demonstration walks the developer through building a simple radar signal processing pipeline, targeted towards detecting objects, with Holoscan. In this example, we generate random radar and waveform data, passing both through:
1. Pulse Compression
2. Moving Target Indication (MTI) Filtering
3. Range-Doppler Map
4. Constant False Alarm Rate (CFAR) Analysis

While this example generates 'offline' complex-valued data, it could be extended to accept streaming data from a phased array system or simulation via modification of the `SignalGeneratorOperator`.

The output of this demonstration is a measure of the number of pulses per second processed on GPU.

 The main objectives of this demonstration are to:
- Highlight developer productivity in building an end-to-end streaming application with Holoscan and existing GPU-Accelerated Python libraries
- Demonstrate how to construct and connect isolated units of work via Holoscan operators, particularly with handling multiple inputs and outputs into an Operator
- Emphasize that operators created for this application can be reused in other ones doing similar tasks

## Building the application
Make sure CMake (https://www.cmake.org) is installed on your system (minimum version 3.20)

- [Holoscan Debian Package](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_dev_deb) - Follow the instructions in the link to install the latest version of Holoscan Debian package from NGC.

- Create a build directory:
  ```bash
  mkdir -p <build_dir> && cd <build_dir>
  ```
- Configure with CMake:

  Make sure CMake can find your installation of the Holoscan SDK. For example, setting `holoscan_ROOT` to its install directory during configuration:

  ```bash
  cmake -S <source_dir> -B <build_dir> -DAPP_simple_radar_pipeline=1 
  ```

  _Notes:_
  _If the error `No CMAKE_CUDA_COMPILER could be found` is encountered, make sure that the :code:`nvcc` executable can be found by adding the CUDA runtime location to your `PATH` variable:_

  ```
  export PATH=$PATH:/usr/local/cuda/bin
  ```

- Build:

  ```bash
  cmake --build <build_dir>
  ```

## Running the application
```bash
<build_dir>/simple_radar_pipeline
```

