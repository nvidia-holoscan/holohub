# V4L2 Video Capture Extension

This GXF extension provides support for USB and HDMI video capture using V4L2.

## Requirements

Install the following dependency:
```sh
sudo apt-get install libv4l-dev=1.18.0-2build1
```

Note that you might not have permissions to open the video devices, run `sudo chmod 666 /dev/video*` to make them available.

## Building the extension

As part of Holohub, running CMake on Holohub and point to Holoscan SDK install tree.
