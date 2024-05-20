# Ultrasound Beamforming with MATLAB GPU Coder

This application does real-time ultrasound beamforming of simulated data. The beamforming algorithm is implemented in MATLAB and converted to CUDA using MATLAB GPU Coder. When the application is run, Holoviz will display the beamformed data in real time.

## Folder Structure

```sh
matlab_beamform
├── data  # Data is generated with generate_data.mlx
│   └── ultrasound_beamforming.bin  # Simulated ultrasound data
├── matlab  # MATLAB files
│   ├── generate_beamform_jetson.m  # MATLAB script to generate CUDA DLLs on Jetson
│   ├── generate_beamform_x86.m  # MATLAB script to generate CUDA DLLs on x86
│   ├── generate_data.mlx  # MATLAB script to generate simulated data
│   └── matlab_beamform.m  # MATLAB function that CUDA code is generated from
├── CMakeLists.txt  # CMake build file
├── main.cpp  # Ultrasound beamforming app
└── matlab_beamform.yaml  # Ultrasound beamforming config
```

## Generate Simulated Data

The required MATLAB Toolboxes are:
* [Phased Array System Toolbox](https://uk.mathworks.com/products/phased-array.html)
* [Communications Toolbox](https://uk.mathworks.com/products/communications.html)

Simply run the script `matlab/generate_data.mlx` from MATLAB and a binary file `ultrasound_beamforming.bin` will be written to a top-level `data` folder. The binary file contains the simulated ultrasound data, prior to beamforming.

## Generate CUDA Code with MATLAB GPU Coder

### x86: Ubuntu

In order to generate the CUDA Code, start MATLAB and `cd` to the `matlab` folder and open the `generate_beamform_x86.m` script. Run the script and a folder `codegen/dll/matlab_beamform` will be generated in the `matlab_beamform` folder.

### arm64: Jetson

On an x86 computer with MATLAB installed, `cd` to the `matlab` folder and open the `generate_beamform_jetson.m` script. Having an `ssh` connection to the Jetson device you want to build the CUDA DLLs on, specify the parameters of that connection in the `hwobj` on line 7, also replace `<ABSOLUTE_PATH>` of `cfg.Hardware.BuildDir` on line 39, as the absolute path (on the Jetson device) to `holohub` folder. Run the script and a folder `MATLAB_ws` will be created in the `matlab_beamform` folder.

## Configure Application

### Configure Holoscan for MATLAB

#### x86: Ubuntu

Define the environment variable:
```sh
export MATLAB_ROOT="/usr/local/MATLAB"
export MATLAB_VERSION="R2023b"
```
where you, if need be, replace `MATLAB_ROOT` with the location of your MATLAB install and `MATLAB_VERSION` with the correct version.

Next, run the HoloHub Docker container:
```sh
./dev_container launch \
    --img nvcr.io/nvidia/clara-holoscan/holoscan:v1.0.3-dgpu \
    --add-volume ${MATLAB_ROOT}/${MATLAB_VERSION} \
    --docker_opts "-e MATLAB_ROOT=/workspace/volumes/${MATLAB_VERSION}"
```

#### arm64: Jetson

The folder `MATLAB_ws`, created by MATLAB, mirrors the folder structure of the host machine and is therefore different from one user to another; hence, we need to specify the path to the `codegen` folder in the `CMakeLists.txt`, in order for the build to find the required libraries. Set the variable `REL_PTH_MATLAB_CODEGEN` to the relative path where the `codegen` folder is located in the `MATLAB_ws` folder. For example, if GPU Coder created the following folder structure on the Jetson device:
```sh
matlab_beamform
└── MATLAB_ws
    └── R2023b
        └── C
            └── Users
                └── Jensen
                    └── holohub
                        └── applications
                            └── matlab_gpu_coder
                                └── matlab_beamform
                                    └── matlab
                                        └── codegen
```
the variable should be set as:
```sh
REL_PTH_MATLAB_CODEGEN=MATLAB_ws/R2023b/C/Users/Jensen/holohub/applications/matlab_gpu_coder/matlab_beamform/matlab/codegen
```