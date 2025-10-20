# Laser Detection

## Overview

This application demonstrates the latency differences between USB and EVT cameras by detecting laser pointer positions on a monitor. It uses two camera sources to track laser positions and displays the results in real-time with different colored icons.

## Hardware Requirements

- USB camera (Logitech 4k Pro Webcam or compatible)
- EVT camera (HB-9000-G 25GE or compatible)
- Monitor with matte screen (120fps or higher refresh rate recommended)
- Safe laser pointer for viewing purposes
- Completed calibration files from both USB and EVT calibration apps

## Setup Instructions

1. Complete the calibration process for both USB and EVT cameras
2. Ensure both cameras are properly connected and configured
3. Position the cameras to have a clear view of the monitor
4. Verify the calibration files (`usb-cali.npy` and `evt-cali.npy`) are present in the build directory

## Running the Application

```bash
[sudo] LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libnvjpeg.so ./holohub run laser_detection
```

## Usage

- A white icon represents the USB camera's laser detection
- A green icon represents the EVT camera's laser detection
- Point the laser at the monitor to see the latency difference between the two cameras
- The icons will move to the coordinates where the laser is detected

## Notes

- Use only a matte screen monitor to avoid specular reflections
- Ensure proper lighting conditions
- Use only safe laser pointers designed for viewing purposes
- If detection is inaccurate, recalibrate both cameras
- The application requires sudo privileges to run
