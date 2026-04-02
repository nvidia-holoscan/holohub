# Atracsys Visualizer

`atracsys_visualizer` is a replay-first HoloHub application for Atracsys visible, infrared,
structured-light, and tracking data. The default mode uses recorded streams so the visualization
stack can be built and exercised without proprietary live-camera dependencies.

The application is split into:
- a required reusable `AtracsysModeSwitcherOp` operator
- an optional `atracsys_camera` operator package for live hardware input
- a single C++ application entrypoint with `replayer` and `live_camera` modes

## Modes

Replay mode:
- replays `visible_base`, `ir_base`, `structured_points`, and `marker_poses`
- demosaics the visible stream
- projects marker geometry overlays
- renders structured-light points and tracking overlays in Holoviz

Live mode:
- enables the optional Atracsys camera source
- produces visible, infrared, marker-pose, and disparity data from hardware
- converts disparity to structured points before visualization

## Container and dependency model

This application includes a custom [Dockerfile](./Dockerfile)
because the live Atracsys path depends on a CUDA-enabled OpenCV build. The public container:
- builds OpenCV 4.10 with CUDA, CUBLAS, and TBB
- exports `OpenCV_DIR=/usr/local/lib/cmake/opencv4`
- keeps replay mode usable without any vendor SDK payloads in the repository

The Atracsys SDK and S3DK remain external dependencies:
- they are not bundled in this repository
- they are not installed by the public Dockerfile
- they must be installed on the host for `--local` builds or added in a private derivative image for containerized live builds

Recommended live SDK layout:
- Atracsys SDK config under `/opt/atracsys-4.9.0/cmake/Atracsys` or pointed to with `Atracsys_DIR`
- S3DK root under `/opt/s3dk` or pointed to with `S3DK_ROOT`

## Build and run

Replay mode:
```bash
./holohub run atracsys_visualizer replayer
```

If you want to prebuild the application container and control the CUDA architecture used for the
OpenCV build, pin the CUDA base explicitly. This matters on machines where HoloHub cannot detect
the NVIDIA driver and falls back to CUDA 13 automatically.
```bash
./holohub build-container atracsys_visualizer \
  --cuda 12 \
  --build-args="--build-arg CUDA_ARCH_BIN=7.5"
```

Live mode follows the usual HoloHub pattern: recurring container settings live in
[metadata.json](/home/artrit/projects/holohub-fork/applications/atracsys_visualizer/metadata.json),
while the public [Dockerfile](./Dockerfile) handles the CUDA-enabled OpenCV build.
If the SDKs are installed at `/opt/atracsys-4.9.0` and `/opt/s3dk`, and the app container has
already been built with the correct CUDA base, the day-to-day live commands stay short:

```bash
./holohub build atracsys_visualizer live_camera
./holohub run atracsys_visualizer live_camera
```

If you prefer a local host/toolchain build instead of the container flow:
```bash
./holohub build atracsys_visualizer live_camera \
  --local \
  --build-with atracsys_camera \
  --configure-args="-DAtracsys_DIR=/opt/atracsys-4.9.0/cmake/Atracsys" \
  --configure-args="-DS3DK_ROOT=/opt/s3dk"
```

The long override-heavy CLI form is still useful for debugging, but it is not the intended steady-state
workflow. In HoloHub, the normal practice is:
- put public, reproducible dependencies in the project Dockerfile
- put recurring mode-specific mounts and environment in `metadata.json`
- use CLI overrides only for one-off troubleshooting

## Controls

- `1`: visible mode
- `2`: infrared mode
- `3`: structured-light mode
- `4`: tracking mode

## Notes

- The application includes a tiny replay-data generator for development and testing.
- For the intended upstream PR, the replay sample should be replaced by a hosted public dataset reference.
- Live mode requires Atracsys hardware, the Atracsys SDK, S3DK, CUDA-enabled OpenCV, and any required USB/container permissions.
