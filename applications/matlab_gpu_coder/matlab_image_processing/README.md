# Image Processing with MATLAB GPU Coder

This application does real-time image processing of Holoscan sample data. The image processing is implemented in MATLAB and converted to CUDA using GPU Coder. When the application is run, Holoviz will display the processed data in real time.

## Folder Structure

```sh
matlab_image_processing
├── matlab  # MATLAB files
│   ├── generate_image_processing_jetson.m  # MATLAB script to generate CUDA DLLs on Jetson
│   ├── generate_image_processing_x86.m  # MATLAB script to generate CUDA DLLs on x86
│   ├── matlab_image_processing.m  # MATLAB function that CUDA code is generated from
│   └── test_image_processing.m  # MATLAB script to test MATLAB function
├── CMakeLists.txt  # CMake build file
├── main.cpp  # Ultrasound beamforming app
└── matlab_image_processing.yaml  # Ultrasound beamforming config
```

## Generate CUDA Code with MATLAB GPU Coder

### x86: Ubuntu

In order to generate the CUDA Code, start MATLAB and `cd` to the `matlab` folder and open the `generate_image_processing_x86.m` script. Run the script and a folder `codegen/dll/matlab_image_processing` will be generated in the `matlab_image_processing` folder.

### arm64: Jetson

On an x86 computer with MATLAB installed, `cd` to the `matlab` folder and open the `generate_image_processing_jetson.m` script. Having an `ssh` connection to the Jetson device you want to build the CUDA DLLs on, specify the parameters of that connection in the `hwobj` on line 7, also replace `<ABSOLUTE_PATH>` of `cfg.Hardware.BuildDir` on line 39, as the absolute path (on the Jetson device) to `holohub` folder. Run the script and a folder `MATLAB_ws` will be created in the `matlab_image_processing` folder.

## Configure Holoscan for MATLAB

If you have not already, start by building HoloHub:
```sh
./dev_container build
```

### x86: Ubuntu

Define the environment variable:
```sh
export MATLAB_ROOT="/usr/local/MATLAB"
export MATLAB_VERSION="R2023b"
```
where you, if need be, replace `MATLAB_ROOT` with the location of your MATLAB install and `MATLAB_VERSION` with the correct version.

Next, run the HoloHub Docker container:
```sh
./dev_container launch \
    --add-volume ${MATLAB_ROOT}/${MATLAB_VERSION} \
    --docker_opts "-e MATLAB_ROOT=/workspace/volumes/${MATLAB_VERSION}"
```
and build the endoscopy tool tracking application to download the necessary data:
```sh
./run build endoscopy_tool_tracking
```

### arm64: Jetson

The folder `MATLAB_ws`, created by MATLAB, mirrors the folder structure of the host machine and is therefore different from one user to another; hence, we need to specify the path to the `codegen` folder in the `CMakeLists.txt`, in order for the build to find the required libraries. Set the variable `REL_PTH_MATLAB_CODEGEN` to the relative path where the `codegen` folder is located in the `MATLAB_ws` folder. For example, if GPU Coder created the following folder structure on the Jetson device:
```sh
matlab_gpu_coder
└── MATLAB_ws
    └── R2023b
        └── C
            └── Users
                └── Jensen
                    └── holohub
                        └── applications
                            └── matlab_gpu_coder
                                └── matlab_image_processing
                                    └── matlab
                                        └── codegen
```
the variable should be set as:
```sh
REL_PTH_MATLAB_CODEGEN=MATLAB_ws/R2023b/C/Users/Jensen/holohub/applications/matlab_gpu_coder/matlab_image_processing/matlab/codegen
```
Next, run the HoloHub Docker container:
```sh
./dev_container launch
```
and build the endoscopy tool tracking application to download the necessary data:
```sh
./run build endoscopy_tool_tracking
```