# XR + Gaussian Splatting

This application demonstrates rendering a 3D scene using Gaussian Splatting in XR.  


### 1. Build the Docker Image

Run the following command in the top-level HoloHub directory:
```bash
./dev_container build --docker_file ./applications/xr_gsplat/Dockerfile --img holohub:xr_gsplat
```
### 2. Install gsplat
```bash
# Inside the container, install gsplat
./dev_container launch --img holohub:xr_gsplat 
pip install git+https://github.com/nerfstudio-project/gsplat.git
```
### 3. Run the application

#### Terminal 1: Launch Container and Start Monado Service
```bash
# If you're already in the container, skip this step
./dev_container launch --img holohub:xr_gsplat

# Inside the container, start the Monado service
monado-service
```
Keep this terminal open and running.

#### Terminal 2: Build and Run the Application
```bash
# Enter the same container (replace <container_id> with actual ID from 'docker ps')
docker exec -it <container_id> bash

# Build the application
./run build xr_gsplat

# Run the application
./run launch xr_gsplat
```