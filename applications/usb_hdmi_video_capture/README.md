# Capture USB and HDMI Video Stream

This app captures either a USB or an HDMI video stream and visualizes it using Holoviz.

## Requirements

Install the following dependency:
```sh
sudo apt-get install libv4l-dev=1.18.0-2build1
```

Note that you might not have permissions to open the video device(s), run:
```sh
 sudo chmod 666 /dev/video?
 ```
 to make them available.

## Parameters

Make sure that the `pixel_format` parameter in the YAML config file is set correctly, options are `RGBA32` and `YUYV`. Also make sure that the `device` parameter is set to the mount point of the device you want to stream from. 

These parameters can be found with:
```sh
v4l2-ctl --list-devices
```
followed by:
```sh
v4l2-ctl -d /dev/video0 --list-formats-ext
```
If you do not have the `v4l2-ctl` app, it can be installed with:
```sh
sudo apt-get install v4l-utils
```

## HDMI IN

The HDMI IN on the dev kit will have to be activated in order for capturing to work. Please look at the relevant dev kit user guide for instructions. Additionally, the parameter `pixel_format` needs to be set to `RGBA32` and  the `device` parameter set to the mount point of the HDMI IN device.

## Run Instructions

First, build the app with the root folder `run.sh` script:
```sh
./run build usb_hdmi_video_capture
```

### C++

```sh
cd build/applications/usb_hdmi_video_capture/cpp
./usb_hdmi_video_capture
```

### Python

```sh
cd applications/usb_hdmi_video_capture/python/
python3 usb_hdmi_video_capture.py
```