# XR + Holoviz

This application demonstrates the integration of Holoscan-XR with Holoviz for extended reality visualization.


## Quick Start Guide

### 1. Build the Docker Image

Run the following command in the top-level HoloHub directory:
```bash
./dev_container build --docker_file ./applications/xr_holoviz/Dockerfile --img holohub:xr_holoviz
```

### 2. Run the application

#### Terminal 1: Launch Container and Start Monado Service
```bash
# Launch the container
./dev_container launch --img holohub:xr_holoviz

# Inside the container, start the Monado service
monado-service
```
Keep this terminal open and running.

#### Terminal 2: Build and Run the Application
```bash
# Enter the same container (replace <container_id> with actual ID from 'docker ps')
docker exec -it <container_id> bash

# Build the application
./run build xr_holoviz

# Run the application
./run launch xr_holoviz
```

#### Set up width and height correctly

The width and height of the HolovizOp should be set to the width and height of the XR display, which can only be obtained during runtime. To set the width and height correctly, we need to:

1. Run the application
2. Find the log showing 
```
XrCompositionLayerManager initialized width: XXX height: YYY
```
3. Copy the width and height
4. Set the width and height of the HolovizOp in `config.yaml`