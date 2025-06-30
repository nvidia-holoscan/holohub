# EVT Camera Calibration

## Overview

This application performs monitor registration using an [Emergent Vision Technologies (EVT)](https://emergentvisiontec.com/) camera. It detects April tags placed at the four corners of a monitor to establish the monitor's position and orientation in 3D space.

## Hardware Requirements

- [EVT HB-9000-G 25GE](https://emergentvisiontec.com/products/bolt-hb-25gige-cameras-rdma-area-scan/hb-9000-g/) camera
- Monitor with April tags at all four corners
- Proper lighting conditions (well-lit environment without backlight)

## Setup Instructions

1. Follow the [Holoscan SDK user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/emergent_setup.html) to set up the EVT camera
2. Place the calibration image with April tags on the monitor
3. Position the camera so it can see all four corners of the monitor
4. Verify camera visibility using the high_speed_endoscopy app

## Running the Application

```bash
./holohub build evt_cam_calibration --local
./holohub run evt_cam_calibration --local --no-local-build
```

## Output

The application generates a calibration file `evt-cali.npy` in the build directory, which contains the monitor's corner coordinates.

## Notes

- The camera must have a clear view of all four April tags
- Avoid backlighting or glare on the monitor
- If using a different camera model, update the camera settings in the Python app or YAML configuration file
- The application requires sudo privileges to run
