# Medical Image Viewer in XR


## Name

Medical Image viewer in XR


## Description

Medical imagery is one of the fastest-growing sources of data in any industry.

We collaborated with Magic Leap on a proof of concept AR viewer for medical imagery built on our Holoscan platform. So when we think about typical diagnostic imaging, x-ray, CT scans, and MRIs come to mind. X-rays are 2D images, so viewing them on a lightbox or, if they’re digital, a computer, is fine. But CT scans and MRIs are 3D. They’re incredibly important technologies, but our way of interacting with them is flawed.This technology helps physicians in so many ways, from training and education to making more accurate diagnoses and ultimately to planning and even delivering more effective treatments.

## Prerequisites 

### OpenXR runtime
OpenXR runtimes are implementations of the OpenXR API that will allow Holoscan XR operators to create XR sessions and render content. 
The runtimes used by Holoscan XR are run as services.You will be able to download the windrunner binaries by running `bash magicleap.sh` inside the `thirdparty/` folder .
```sh
cd applications/volume_rendering_xr
bash magicleap.sh`
```

### Android Tools
Android Debug Bridge (ADB) is a command-line tool that allows you to communicate with Android Open Source Project (AOSP) devices such as the Magic Leap 2 .

* First you need to install Android Tools

`sudo apt install android-tools-adb`

* You will need to connect to the IP of your Head Mounted Display (HMD).  While wearing the HMD, go to "Settings->About" menu page to find the IP address of the HMD.

`adb connect <IP_headset>`

Upon successful adb connect, you should be able to see the devices listed with

`adb devices`

You will need to pair the IGX devkit and the headset by running the following from `applications/volume_rendering_xr` folder

`./thirdparty/magicleap/windrunner-aarch64/bin/SetupRemoteViewerIP -i <IP_devkit>`

## Development

### Setup VSCode Dev Container Environment

Install VSCode for [Arm64](https://code.visualstudio.com/download#) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) with Docker.

The `volume_rendering_xr` app is built on top of vscode. In order to build the app, please copy the `.devcontainer` and `.vscode` folders to the root directory of your cloned holohub repository.

When using the Dev Container the first time:
* Start VSCode and go to `View -> Command Palette -> Dev Containers: Rebuild Container Without Cache`. This will build the container and it could take a few minutes.

### Start an OpenXR runtime service


The Magic Leap Remote OpenXR runtime (windrunner) is the default configured by
the dev container.

For rapid iteration without a Magic Leap device, The Monado OpenXR runtime
provides a "Simulated HMD" mode that does not require specialized XR hardware to
use (though it does not support any user inputs).

#### Option A: Start the Magic Leap Remote OpenXR runtime service (Windrunner)

From a terminal _inside_ the dev container, ensure `openxr_windrunner.json` is
set as the active runtime (default):

`sudo update-alternatives --config openxr1-active-runtime`

Then start the service

```
windrunner-service
```

#### Option B: Start the Monado OpenXR runtime service

From a terminal _inside_ the dev container, ensure that `openxr_monado.json` is
set as the active runtime:

```
sudo update-alternatives --config openxr1-active-runtime
```

Then start the service:

```
monado-service
```

NOTE: If you switch back to the Magic Leap runtime, don't forget to update the
active runtime alternative with `update-alternatives` (above).

### Build and Run the XR Volume Renderer Application

Ensure an OpenXR runtime service is running, and the correct runtime is set as
active (above).

To build the app, you need to run

`./run build volume_rendering_xr`

To launch the app

`./run launch volume_rendering_xr`

With the command above, the path of the volume configuration file, volume density file and volume mask file will be passed to the application. You can also see how to manually run the application by going to the build folder and running

`./applications/volume_rendering_xr/volume_rendering_xr -h`





