# USB Camera Calibration

## Overview

This application performs monitor registration using a USB camera. It detects [April tags](https://github.com/AprilRobotics/apriltag) placed at the four corners of a monitor to establish the monitor's position and orientation in 3D space.

## Hardware Requirements

- [Logitech 4k Pro Webcam](https://www.logitech.com/en-us/products/webcams/4kprowebcam.960-001390.html) or compatible USB camera
- Monitor with April tags at all four corners
- Proper lighting conditions (well-lit environment without backlight)

## Setup Instructions

1. Ensure the USB camera is properly connected
2. Place the calibration image with April tags on the monitor
3. Position the camera so it can see all four corners of the monitor
4. Verify camera visibility using the v4l2_camera app

## Running the Application

```bash
[sudo] LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libnvjpeg.so ./holohub run usb_cam_calibration
```

## Output

The application generates a calibration file `usb-cali.npy` in the build directory, which contains the monitor's corner coordinates.

## Notes

- The camera must have a clear view of all four April tags
- Avoid backlighting or glare on the monitor
- If using a different camera model, update the camera settings in the Python app or YAML configuration file
