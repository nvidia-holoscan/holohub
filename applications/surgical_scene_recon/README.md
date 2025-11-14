# G-SHARP: Gaussian Surgical Edge-Hardware Accelerated Real-time Pipeline

Real-time 3D surgical scene reconstruction using Gaussian Splatting in a Holoscan streaming pipeline with temporal deformation for accurate tissue modeling. 

---

## Overview

This application combines **Holoscan SDK** for real-time streaming, **3D Gaussian Splatting** for neural 3D reconstruction, and **temporal deformation networks** for accurate surgical scene modeling.

### Architecture

The application provides a complete end-to-end pipeline for surgical scene reconstruction:

#### **1. Data Accumulation & Training Pipeline**

The training pipeline processes endoscopic surgical video frames to learn a 3D representation:

- **Multi-modal Data Loading:** Accumulates RGB images, stereo depth maps, and tool segmentation masks from the EndoNeRF dataset
- **Multi-frame Point Cloud Initialization:** Aggregates depth information across all frames to create a dense initial 3D point cloud (~30k points vs. ~3k from single frame)
- **Gaussian Splatting Training:** Optimizes 3D Gaussians (position, scale, rotation, opacity, color) to represent the surgical scene
- **Temporal Deformation Network:** Learns tissue deformation over time using a 4D spatiotemporal HexPlane field with MLP-based deformation
- **Tool Removal:** Uses segmentation masks to reconstruct tissue-only scenes, removing surgical instruments

**Training Output:** A checkpoint containing:
- Base 3D Gaussian parameters (means, scales, quaternions, opacities, spherical harmonics)
- Deformation network weights (for temporal modeling)
- Scene metadata (camera parameters, normalization scales)

#### **2. Real-time Rendering Pipeline**

The rendering pipeline uses Holoscan operators for real-time visualization:

```
EndoNeRFLoaderOp → GsplatLoaderOp → GsplatRenderOp → HolovizOp
                                                    ↓
                                              ImageSaverOp
```

**Pipeline Components:**
- **EndoNeRFLoaderOp:** Streams camera poses and timestamps frame-by-frame
- **GsplatLoaderOp:** Loads trained checkpoint and deformation network (once at startup)
- **GsplatRenderOp:** Applies temporal deformation for current frame time, then renders using differentiable rasterization
- **HolovizOp:** Real-time visualization with Holoscan's GPU-accelerated display
- **ImageSaverOp:** Saves rendered frames to disk for inspection

#### **3. Operating Modes**

- **Dynamic Rendering (Recommended):** Applies temporal deformation to accurately reconstruct tissue as it deforms over time - highest quality
- **Static Rendering (Fast Preview):** Uses base Gaussians without deformation - faster but less accurate for moving tissue
- **Training Mode:** Trains a new model on your dataset - enables custom scene reconstruction

---

## Dataset

### EndoNeRF Dataset

This application uses the **EndoNeRF dataset** for surgical scene reconstruction. 

**Download and Setup:**
1. Visit the EndoNeRF repository: https://github.com/med-air/EndoNeRF
2. Follow their instructions to download and process the dataset
3. This application uses the `pulling_soft_tissues` clip (one of the accessible clips)

**Dataset Structure:**
After processing, your dataset should have the following structure:
```
<DATA_PATH>/
└── EndoNeRF/
    └── pulling/
        ├── images/           # RGB frames
        ├── depth/            # Depth maps
        ├── masks/            # Tool masks
        └── poses.txt         # Camera poses
```

**Path Configuration:**
You'll need to set two environment variables for your system:
- `HOLOHUB_PATH`: Path to your holohub-internal repository
- `DATA_PATH`: Path to where you've downloaded and extracted the EndoNeRF dataset

**Example Setup:**
```bash
export HOLOHUB_PATH=/path/to/holohub-internal
export DATA_PATH=/path/to/your/datasets
```

After setup, your EndoNeRF dataset should be accessible at:
```
${DATA_PATH}/EndoNeRF/pulling/
```

---

## Complete Setup Guide

### Step-by-Step Setup

1. **Set Environment Variables**
   ```bash
   export HOLOHUB_PATH=/path/to/holohub-internal
   export DATA_PATH=/path/to/your/datasets
   ```

2. **Download EndoNeRF Dataset**
   - Visit: https://github.com/med-air/EndoNeRF
   - Download and extract the `pulling_soft_tissues` dataset
   - Place it in: `${DATA_PATH}/EndoNeRF/pulling/`

3. **Build Docker Image** (one-time, ~10-15 minutes)
   ```bash
   cd ${HOLOHUB_PATH}
   docker build -f applications/surgical_scene_recon/Dockerfile -t surgical_scene_recon:latest .
   ```

4. **Run Complete Workflow** (RECOMMENDED - automated script)
   ```bash
   cd ${HOLOHUB_PATH}/applications/surgical_scene_recon
   ./run_complete_docker.sh production
   ```
   
   This automated script handles training + rendering in one command!
   See [Complete Workflow Script](#complete-workflow-script-automated) section for details.

   **OR** manually train and then run inference separately:
   
   a. Train the model:
   ```bash
   cd ${HOLOHUB_PATH}
   docker run --rm --gpus all \
     -v ${HOLOHUB_PATH}:/workspace/holohub \
     -v ${DATA_PATH}:/workspace/data \
     -w /workspace/holohub/applications/surgical_scene_recon \
     surgical_scene_recon:latest \
     python run_surgical_recon.py \
       --mode train \
       --data_dir /workspace/data/EndoNeRF/pulling \
       --output_dir output/production_training \
       --training_iterations 2000 \
       --coarse_iterations 200
   ```
   
   b. Run production inference:
   ```bash
   cd ${HOLOHUB_PATH}
   docker run --rm --gpus all \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v ${HOLOHUB_PATH}:/workspace/holohub \
     -v ${DATA_PATH}:/workspace/data \
     -w /workspace/holohub/applications/surgical_scene_recon \
     surgical_scene_recon:latest \
     python run_surgical_recon.py \
       --mode inference \
       --data_dir /workspace/data/EndoNeRF/pulling \
       --checkpoint /workspace/holohub/applications/surgical_scene_recon/output/production_training/ckpts/fine_best_psnr.pt
   ```

---

## Key Features

- ✅ **Real-time Visualization:** Stream surgical scene reconstruction at >10 FPS
- ✅ **Temporal Deformation:** Accurate per-frame tissue modeling
- ✅ **Tool Removal:** Tissue-only reconstruction mode (instruments excluded)
- ✅ **Two Operation Modes:** Inference-only OR Train-then-render 🆕
- ✅ **Integrated Training:** On-the-fly model training from streamed data 🆕
- ✅ **Two Rendering Modes:** Static (fast) or Dynamic (high-quality)
- ✅ **Holoviz Integration:** Interactive 3D visualization
- ✅ **Production Ready:** Tested and optimized pipeline

### What It Does

**Input:** EndoNeRF surgical dataset (RGB images + camera poses + tool masks)  
**Process:** Gaussian Splatting with temporal deformation network  
**Output:** Real-time 3D tissue reconstruction without surgical instruments

### Operation Modes

1. **Inference-Only Mode:** Use pre-trained checkpoint for immediate rendering
2. **Train-Then-Render Mode:** 🆕 Collect data → Train model → Render with trained checkpoint

---

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU (RTX 3000+ recommended)
- X11 display server
- EndoNeRF dataset (see [Dataset](#dataset) section)
- Path environment variables set: `HOLOHUB_PATH` and `DATA_PATH`

### Step 1: Build the Docker Image

Build the surgical scene reconstruction Docker image (one-time setup, ~10-15 minutes):

```bash
cd ${HOLOHUB_PATH}

# Standard build
docker build \
  -f applications/surgical_scene_recon/Dockerfile \
  -t surgical_scene_recon:latest \
  .
```

**Custom base image (optional):**
```bash
# Use a different Holoscan version or custom base
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v3.8.0-cuda12-dgpu \
  -f applications/surgical_scene_recon/Dockerfile \
  -t surgical_scene_recon:latest \
  .
```

**Rebuild (if updating):**
```bash
# Remove old image first
docker rmi surgical_scene_recon:latest

# Then rebuild
docker build -f applications/surgical_scene_recon/Dockerfile -t surgical_scene_recon:latest .
```

**Verify the build:**
```bash
docker images surgical_scene_recon:latest
```

You should see output similar to:
```
REPOSITORY                TAG       IMAGE ID       CREATED          SIZE
surgical_scene_recon      latest    abc123def456   2 minutes ago    26GB
```

**What's in the image:**
- Holoscan SDK 3.7.0 with CUDA 12.x support
- PyTorch ≥2.1.0, torchvision ≥0.16.0
- gsplat ≥1.0.0 for Gaussian Splatting
- Training dependencies: tqdm, pyyaml, tensorboard, lpips, torchmetrics
- 3D processing: open3d, plyfile, fpsample
- Image processing: pillow, imageio, opencv-python
- Scientific: numpy, scipy, cupy-cuda12x, scikit-learn

### Step 2: Run Production Inference

Once the Docker image is built, run a full production inference with pre-trained checkpoint:

```bash
cd ${HOLOHUB_PATH}

docker run --rm --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${HOLOHUB_PATH}:/workspace/holohub \
  -v ${DATA_PATH}:/workspace/data \
  -w /workspace/holohub/applications/surgical_scene_recon \
  surgical_scene_recon:latest \
  python run_surgical_recon.py \
    --mode inference \
    --data_dir /workspace/data/EndoNeRF/pulling \
    --checkpoint /workspace/data/checkpoints/fine_best_psnr.pt
```

**Note:** You'll need a pre-trained checkpoint. To train your own, see [Training Mode](#training-mode) below.

**Press ESC or close window to exit visualization.**

---

## Training Mode

If you don't have a pre-trained checkpoint, you can train your own model from the EndoNeRF dataset.

### Quick Training Test

Train a quick test model (30 frames, 500 iterations, ~15 minutes):

```bash
cd ${HOLOHUB_PATH}

docker run --rm --gpus all \
  -v ${HOLOHUB_PATH}:/workspace/holohub \
  -v ${DATA_PATH}:/workspace/data \
  -w /workspace/holohub/applications/surgical_scene_recon \
  surgical_scene_recon:latest \
  python train_standalone.py \
    --data_dir /workspace/data/EndoNeRF/pulling \
    --output_dir output/quick_test \
    --training_iterations 500 \
    --coarse_iterations 50 \
    --num_frames 30
```

### Full Production Training

Train a production-quality model (all 63 frames, 2000 iterations, ~25 minutes):

```bash
cd ${HOLOHUB_PATH}

docker run --rm --gpus all \
  -v ${HOLOHUB_PATH}:/workspace/holohub \
  -v ${DATA_PATH}:/workspace/data \
  -w /workspace/holohub/applications/surgical_scene_recon \
  surgical_scene_recon:latest \
  python run_surgical_recon.py \
    --mode train \
    --data_dir /workspace/data/EndoNeRF/pulling \
    --output_dir output/production_training \
    --training_iterations 2000 \
    --coarse_iterations 200
```

**Output:** Trained checkpoints will be saved to `${HOLOHUB_PATH}/applications/surgical_scene_recon/output/` directory.

**Next Step:** Use the trained checkpoint for inference by replacing the `--checkpoint` path in the production inference command above.

---

## Complete Workflow Script (Automated)

For convenience, a complete automated workflow script is provided that handles training and rendering in a single command.

### `run_complete_docker.sh` - One-Command Workflow

This script automates the entire pipeline:
1. ✅ Data accumulation from EndoNeRF dataset
2. ✅ Model training with Gaussian Splatting
3. ✅ Real-time rendering with trained checkpoint

**Prerequisites:**
- Docker image built: `surgical_scene_recon:latest`
- Environment variables set: `HOLOHUB_PATH` and `DATA_PATH`
- EndoNeRF dataset downloaded

### Usage

Navigate to the application directory and run:

```bash
cd ${HOLOHUB_PATH}/applications/surgical_scene_recon

# Quick test mode (30 frames, 500 iterations, ~15 minutes)
./run_complete_docker.sh quick

# Production mode (all 63 frames, 2000 iterations, ~30 minutes)
./run_complete_docker.sh production
```

### Modes

| Mode | Frames | Iterations | Time | Use Case |
|------|--------|-----------|------|----------|
| **quick** | 30 | 500 | ~15 min | Testing, development |
| **production** | 63 (all) | 2000 | ~30 min | Production quality, final results |

### What Happens

**Stage 1 & 2: Training**
- Loads EndoNeRF dataset (images, poses, masks)
- Trains Gaussian Splatting model with temporal deformation
- Saves checkpoint to `output/docker_complete_[mode]/trained_model/ckpts/`

**Stage 3: Rendering**
- Automatically loads trained checkpoint
- Renders all frames with real-time visualization
- Displays in Holoviz window (3-minute auto-timeout)
- Press ESC to exit early

### Output

After completion, you'll have:
```
applications/surgical_scene_recon/
└── output/
    └── docker_complete_[mode]/
        ├── training_ingestion/         # Accumulated training data
        └── trained_model/
            └── ckpts/
                └── fine_best_psnr.pt   # Your trained checkpoint
```

### Example: Quick Test

Complete end-to-end workflow in ~15 minutes:

```bash
# Set paths (if not already set)
export HOLOHUB_PATH=/path/to/holohub-internal
export DATA_PATH=/path/to/your/datasets

# Navigate and run
cd ${HOLOHUB_PATH}/applications/surgical_scene_recon
./run_complete_docker.sh quick
```

This will:
- Train on first 30 frames (~10 min)
- Render results (~3 min visualization)
- Save checkpoint for later use

### Example: Production Run

Full production workflow in ~30 minutes:

```bash
cd ${HOLOHUB_PATH}/applications/surgical_scene_recon
./run_complete_docker.sh production
```

This will:
- Train on all 63 frames (~25 min)
- Render with best quality (~3 min visualization)
- Create production-ready checkpoint

### Reusing the Trained Checkpoint

After the script completes, you can reuse the trained checkpoint:

```bash
cd ${HOLOHUB_PATH}

docker run --rm --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${HOLOHUB_PATH}:/workspace/holohub \
  -v ${DATA_PATH}:/workspace/data \
  -w /workspace/holohub/applications/surgical_scene_recon \
  surgical_scene_recon:latest \
  python run_surgical_recon.py \
    --mode inference \
    --data_dir /workspace/data/EndoNeRF/pulling \
    --checkpoint /workspace/holohub/applications/surgical_scene_recon/output/docker_complete_production/trained_model/ckpts/fine_best_psnr.pt
```

---

## Architecture

### Pipeline Flow

```
EndoNeRFLoader → GsplatLoader → GsplatRender → Holoviz
  (RGB + poses      (checkpoint     (temporal        (display)
   + time)           + deform_net)    deformation)
```

### Two Rendering Modes

| Mode | Description | Use Case | Performance |
|------|-------------|----------|-------------|
| **Static** | Averaged reconstruction | Quick preview, testing | >20 FPS |
| **Dynamic** | Per-frame deformation | Production, analysis | >10 FPS |

### Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `EndoNeRFLoaderOp` | Load EndoNeRF dataset | ✅ Complete |
| `GsplatLoaderOp` | Load checkpoint & deform network | ✅ Complete |
| `GsplatRenderOp` | Render with temporal deformation | ✅ Complete |
| `HolovizOp` | Real-time visualization | ✅ Complete |
| `ImageSaverOp` | Save frames (optional) | ✅ Complete |

---

## How It Works

### 1. Data Loading
- Loads RGB images, depth maps, and tool masks from EndoNeRF dataset
- Extracts camera poses and intrinsics
- Computes temporal values (normalized 0-1 across sequence)

### 2. Checkpoint Loading
- **Static Mode:** Load gaussians, apply activations (exp, sigmoid)
- **Dynamic Mode:** Load base gaussians + deformation network (PyTorch model)

### 3. Rendering
- **Static Mode:** Direct rendering with pre-activated parameters
- **Dynamic Mode:** 
  1. Apply deformation network based on time value
  2. Activate deformed parameters (exp, sigmoid)
  3. Render with gsplat

### 4. Visualization
- Display in Holoviz window
- Real-time playback through all frames
- Loop continuously or single pass

---

## Technical Details

### Temporal Deformation Network

The deformation network models tissue deformation and tool movement over time:

- **Input:** Base gaussian parameters + time value (0-1)
- **Architecture:** HexPlane 4D grid + MLPs (position, scale, rotation, opacity)
- **Output:** Deformed gaussian parameters for that specific time
- **Inference:** Direct PyTorch (no conversion, full precision)

### Gaussian Splatting

- **Renderer:** gsplat library (CUDA-accelerated)
- **Splats:** ~48,000 gaussians
- **SH Degree:** 3 (16 coefficients per gaussian)
- **Resolution:** 640×512 pixels
- **Format:** RGB (3 channels)

### Dataset

- **Source:** EndoNeRF "pulling" dataset
- **Frames:** 63 training views
- **Modality:** RGB + Depth + Tool Masks
- **Camera:** Calibrated endoscope

---

## Performance

### Tested Configuration
- **GPU:** NVIDIA RTX 6000 Ada Generation
- **Container:** Holoscan SDK 3.7.0
- **Gaussians:** 48,758 splats


## File Structure

```
surgical_scene_recon/
├── README.md                           # This file - Complete documentation
├── Dockerfile                          # Production container
│
├── run_complete_docker.sh              # Complete workflow (automated)
├── run_surgical_recon.py               # Unified Python interface
├── train_standalone.py                 # Standalone training script
├── run_complete_workflow.py            # Python complete workflow
│
├── operators/                          # Custom operators
│   ├── endonerf_loader_op.py          # Data loading
│   ├── gsplat_loader_op.py            # Checkpoint loading
│   ├── gsplat_render_op.py            # Rendering
│   ├── image_saver_op.py              # Image saving
│
├── training/                           # 🆕 Training code
│   ├── gsplat_train.py                # Main training script
│   ├── scene/                         # Data loaders
│   └── utils/                         # Training utilities
│
├── tests/                              # Test scripts
│   ├── test_dynamic_rendering_viz.py  # Dynamic mode test
│   ├── test_static_rendering_viz.py   # Static mode test
│   ├── test_data_loading.py           # Data loading test
│   ├── test_minimal_holoviz.py        # Minimal Holoviz test
│   └── run_tests.sh                   # Automated test suite

```

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (RTX 3000+ recommended)
- 16GB+ GPU memory
- X11 display server

### Software
- Docker with NVIDIA Container Toolkit
- Holoscan SDK 3.7.0 (in container)
- Python 3.10+
- CUDA 12.x

### Dependencies (Auto-installed)
- PyTorch ≥2.1.0
- gsplat ≥1.0.0
- CuPy (CUDA 12.x)
- scipy, pillow

---

### ✅ Phase 1

1. **Data Loading** - EndoNeRF dataset support
2. **Static Rendering** - Fast, averaged reconstruction
3. **Dynamic Rendering** - Per-frame temporal deformation
4. **Holoviz Visualization** - Real-time interactive display
5. **Image Saving** - Optional frame export
6. **Production Quality** - Tested with 20,000+ frames

### Technical Highlights

- Direct PyTorch inference (no conversions)
- Temporal deformation with HexPlane 4D grids
- Tool removal via training-time masking
- Real-time performance on modern GPUs
- Clean Holoscan operator architecture

---

## Use Cases

### Surgical Review & Training
- Visualize anatomy without instrument occlusion
- Review tissue deformation over surgical procedure
- Training tool for surgical techniques

### Research & Development
- 3D reconstruction quality analysis
- Temporal dynamics study
- Novel view synthesis

### Interactive Exploration
- View surgical scene from any angle
- Understand spatial relationships
- Analyze tissue deformation patterns

---

## Comparison: Static vs Dynamic

### When to Use Static Mode
- ✅ Quick previews and testing
- ✅ When temporal accuracy not critical
- ✅ Faster iteration during development
- ✅ Lower computational requirements

### When to Use Dynamic Mode
- ✅ Production visualization
- ✅ Accurate tissue deformation modeling
- ✅ Research and analysis
- ✅ Highest visual quality

### Quality Comparison
- **Static:** Good reconstruction, averaged over time
- **Dynamic:** Excellent reconstruction, per-frame accurate
- **Deformation:** Dynamic captures mean tissue displacement of ~0.71 units

---

## License

This application combines multiple components with different licenses:
- Holoscan SDK (Apache 2.0)
- gsplat (Apache 2.0)
- Surgical Gaussian Training Code (Research License)
