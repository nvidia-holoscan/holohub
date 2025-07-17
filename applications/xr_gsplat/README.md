# XR + Gaussian Splatting

This application demonstrates rendering a 3D scene using Gaussian Splatting in XR.  
We provide setup steps to run and view the scene with the default [Monado](https://monado.dev/) [OpenXR](https://www.khronos.org/openxr/) simulator below. Users with an OpenXR compatible headset may bring their own OpenXR runtime for XR viewing.
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
pip install -r requirements.txt
# Download the data
python datasets/download_dataset.py
```

#### 0.3. Train the model
```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden
```

__note__: Training time is observed to take about 30 minutes on Intel i9 CPU + NVIDIA RTX A5000 dGPU

#### 0.4. Set up the checkpoint paths in `config.yaml`



### 1. Run the application

#### Terminal 1: Launch Container and Start Monado Service
```bash
# If you're already in the container, skip this step
./holohub run-container xr_gsplat

# Inside the container, start the Monado service
monado-service
```
Keep this terminal open and running.

#### Terminal 2: Build and Run the Application
```bash
# Enter the same container (replace <container_id> with actual ID from 'docker ps')
docker exec -it <container_id> bash

# Build and run the application
./holohub run xr_gsplat
```