# XR + Gaussian Splatting

This application demonstrates rendering a 3D scene using Gaussian Splatting in XR.  

### 0. Training a Gaussian Splatting Model
The below instructions are based on the [gsplat colmap example](https://docs.gsplat.studio/main/examples/colmap.html).

#### 0.1. Clone the gsplat repo
```bash
git clone https://github.com/nerfstudio-project/gsplat.git
```

#### 0.2. Install dependencies and download the data
```bash
cd gsplat/examples
# Install torch
pip install torch
# Install gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git
# Install dependencies
pip install -r require  ments.txt
# Download the data
python datasets/download_dataset.py
```

#### 0.3. Train the model
```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden
```

#### 0.4. Set up the checkpoint paths in `config.yaml`


### 1. Build the Docker Image

Run the following command in the top-level HoloHub directory:
```bash
./dev_container build --docker_file ./applications/xr_gsplat/Dockerfile --img holohub:xr_gsplat
```
### 2. Install gsplat
```bash
# Inside the container, install gsplat from source
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