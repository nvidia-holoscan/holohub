# V4L2 Plus Extension

This GXF extension provides support for USB and HDMI input on the Holoscan dev kits.

## Requirements

Install the following two dependencies:
```sh
sudo apt-get install ffmpeg=7:4.2.7-0ubuntu0.1
sudo apt-get install libv4l-dev=1.18.0-2build1
```

Note that you might not have permissions to open the video devices, run `sudo chmod 666 /dev/video*` to make them available.

## Building the extension

As part of Holohub, running CMake on Holohub and point to Holoscan SDK install tree.
