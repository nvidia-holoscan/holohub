# G-SHARP: Gaussian Surgical Edge-Hardware Accelerated Real-time Pipeline

Real-time 3D surgical scene reconstruction using Gaussian Splatting in a Holoscan streaming pipeline with temporal deformation for accurate tissue modeling.

---

## Overview

This application demonstrates real-time 3D surgical scene reconstruction by combining **Holoscan SDK** for high-performance streaming, **3D Gaussian Splatting** for neural 3D representation, and **temporal deformation networks** for accurate modeling of dynamic tissue.

The application provides a complete end-to-end pipeline - from raw surgical video to real-time 3D reconstruction - enabling researchers and developers to train custom models on their own endoscopic data and visualize results with GPU-accelerated rendering.

![Surgical Scene Reconstruction Demo](surg_recon_inference.gif)

**Key Features:**
- ✅ **Real-time Visualization:** Stream surgical scene reconstruction at >30 FPS using Holoscan
- ✅ **Temporal Deformation:** Accurate per-frame tissue modeling as it deforms over time
- ✅ **Tool Removal:** Tissue-only reconstruction mode (surgical instruments automatically excluded)
- ✅ **End-to-End Training:** On-the-fly model training from streamed endoscopic data
- ✅ **Two Operation Modes:** Inference-only (with pre-trained checkpoint) OR Train-then-render
- ✅ **Production Ready:** Tested and optimized Holoscan pipeline with complete Docker containerization

**What It Does:**
- **Input:** EndoNeRF surgical dataset (RGB images + stereo depth + camera poses + tool masks)
- **Process:** Multi-frame Gaussian Splatting with 4D spatiotemporal deformation network
- **Output:** Real-time 3D tissue reconstruction without surgical instruments

**Use Cases:**
- Surgical scene understanding and visualization
- Tool-free tissue reconstruction for analysis
- Research in surgical vision and 3D reconstruction
- Development of real-time surgical guidance systems

---

## Architecture

The application provides a complete end-to-end pipeline:

### Training Pipeline

Processes endoscopic video to learn 3D representation:

- **Multi-modal Data Loading:** RGB images, depth maps, tool segmentation masks
- **Multi-frame Point Cloud Init:** Dense initialization (~30k points from all frames)
- **Gaussian Splatting Training:** Optimizes 3D Gaussians (position, scale, rotation, opacity, color)
- **Temporal Deformation Network:** 4D spatiotemporal HexPlane field with MLP-based deformation
- **Tool Removal:** Uses segmentation masks for tissue-only reconstruction

![Training Visualization - Ground Truth vs Rendered](train_gt_animation.gif)

### Real-time Rendering Pipeline

Holoscan operators for real-time visualization:

```
EndoNeRFLoaderOp → GsplatLoaderOp → GsplatRenderOp → HolovizOp
                                                    ↓
                                              ImageSaverOp
```

**Components:**
- **EndoNeRFLoaderOp:** Streams camera poses and timestamps
- **GsplatLoaderOp:** Loads checkpoint and deformation network
- **GsplatRenderOp:** Applies temporal deformation and renders
- **HolovizOp:** Real-time GPU-accelerated visualization

---

## Dataset

### EndoNeRF Dataset Setup

This application uses the **EndoNeRF "pulling_soft_tissues" dataset**.

#### Download

**Direct Google Drive:** https://drive.google.com/drive/folders/1zTcX80c1yrbntY9c6-EK2W2UVESVEug8?usp=sharing

In the Google Drive folder, you'll see:
- `cutting_tissues_twice`
- `pulling_soft_tissues` ← **Download this one**

**Alternative:** Visit the [EndoNeRF repository](https://github.com/med-air/EndoNeRF)

#### Setup

Place the dataset at the correct location:

```bash
# Create data directory
mkdir -p data/EndoNeRF

# Extract and move downloaded dataset
mv ~/Downloads/pulling_soft_tissues data/EndoNeRF/pulling

# Or copy if extracted elsewhere
cp -r /path/to/pulling_soft_tissues data/EndoNeRF/pulling
```

**⚠️ Critical:** Dataset MUST be physically at `holohub/data/EndoNeRF/pulling/` - do NOT use symlinks! Docker containers cannot follow symlinks outside mounted volumes.

#### Verify

Your dataset should have this structure:

```
holohub/
└── data/
    └── EndoNeRF/
        └── pulling/
            ├── images/              # 63 RGB frames (.png)
            ├── depth/               # 63 depth maps (.png)
            ├── masks/               # 63 tool masks (.png)
            └── poses_bounds.npy     # Camera poses (8.5 KB)
```

**Verification command:**
```bash
ls data/EndoNeRF/pulling/
# Should show: images  depth  masks  poses_bounds.npy
```

---

## Quick Start

### Prerequisites

- NVIDIA GPU (RTX 3000+ series recommended)
- Docker with NVIDIA GPU support
- X11 display server
- ~2 GB free disk space (dataset)
- ~30 GB free disk space (Docker container)

### Three Simple Steps

#### Step 1: Clone HoloHub

```bash
git clone https://github.com/nvidia-holoscan/holohub.git
cd holohub
```

#### Step 2: Download and Place Dataset

```bash
# Download from Google Drive (link above in Dataset section)
# Get the 'pulling_soft_tissues' folder

# Create directory and place dataset
mkdir -p data/EndoNeRF
mv ~/Downloads/pulling_soft_tissues data/EndoNeRF/pulling

# Verify
ls data/EndoNeRF/pulling/
# Should show: images  depth  masks  poses_bounds.npy
```

#### Step 3: Run Training

**For testing/verification:**
```bash
./holohub run surgical_scene_recon training_quick
```

**⚠️ Note:** `training_quick` produces **static/jittery results** - this is expected! It's for testing only:
- ✅ Verifies your setup works (~10 min training)
- ❌ NOT for production quality

**For production quality:**
```bash
./holohub run surgical_scene_recon training_full
```

**What happens:**
1. **First time:** Builds Docker container (~10-15 minutes, one-time)
2. **Training:** 
   - `training_quick`: 30 frames, 500 iterations (~10 min) → Testing only
   - `training_full`: 63 frames, 2000 iterations (~30 min) → Production quality ⭐
3. **Auto-visualization:** Holoviz window opens showing your reconstruction

### Available Modes

Check all modes:
```bash
./holohub modes surgical_scene_recon
```

**Three modes available:**

| Mode | Description | Time | Quality |
|------|-------------|------|---------|
| `training_quick` | Testing/verification only | ~10 min | Lower (static/jittery) |
| `training_full` ⭐ | Production quality | ~30 min | High (smooth deformation) |
| `dynamic_rendering` | Render with trained checkpoint | ~2 min | Uses your trained model |

### Rendering with Trained Model

After training completes, render with your checkpoint:

```bash
./holohub run surgical_scene_recon dynamic_rendering
```

Checkpoint location: `applications/surgical_scene_recon/output/trained_model/ckpts/fine_best_psnr.pt`

---

## Advanced Usage

### Troubleshooting

**Problem: "FileNotFoundError: poses_bounds.npy not found"**
- **Cause:** Dataset not in correct location or symlink used
- **Solution:** Ensure dataset is physically at `holohub/data/EndoNeRF/pulling/`
- **Verify:** Run `file data/EndoNeRF` - should show "directory", not "symbolic link"

**Problem: "Unable to find image holohub-surgical_scene_recon"**
- **Cause:** Container not built yet
- **Solution:** Remove `--no-docker-build` flag (let CLI build automatically)
- **Or:** Manually build: `./holohub build-container surgical_scene_recon`

**Problem: Holoviz window doesn't appear**
- **Cause:** X11 forwarding not enabled
- **Solution:** Run `xhost +local:docker` before training
- **Verify:** Check `echo $DISPLAY` shows a value

**Problem: GPU out of memory**
- **Cause:** Another process using GPU
- **Solution:** Check `nvidia-smi` and stop other processes
- **Or:** Reduce batch size (advanced - edit training config)

### Container Management

**Build container separately:**
```bash
./holohub build-container surgical_scene_recon
```

**Run container (builds if needed):**
```bash
./holohub run-container surgical_scene_recon
```

**Custom base image:**
```bash
./holohub build-container surgical_scene_recon \
  --base-img nvcr.io/nvidia/clara-holoscan/holoscan:v3.8.0-cuda12-dgpu
```

### Training Outputs

After training, find your results:

```bash
ls applications/surgical_scene_recon/output/trained_model/
```

**Directory structure:**
- `ckpts/` - Model checkpoints
  - `fine_best_psnr.pt` - Best checkpoint (use this for rendering)
  - `fine_step00XXX.pt` - Regular checkpoints
- `logs/` - Training logs
- `tb_logs/` - TensorBoard logs
- `renders/` - Saved render frames

**View training logs:**
```bash
tensorboard --logdir applications/surgical_scene_recon/output/trained_model/tb_logs
```

### CLI Help

**Get mode information:**
```bash
./holohub modes surgical_scene_recon
```

**Get run help:**
```bash
./holohub run surgical_scene_recon -h
./holohub run -h
```

**Get build help:**
```bash
./holohub build-container -h
```

---

## Technical Details

### Pipeline Architecture

**Training (gsplat_train.py):**
1. **Data Loading:** EndoNeRF parser loads RGB, depth, masks, poses
2. **Initialization:** Multi-frame point cloud (~30k points)
3. **Two-Stage Training:**
   - **Coarse:** Learn base Gaussians (no deformation)
   - **Fine:** Add temporal deformation network
4. **Optimization:** Adam with batch-size scaled learning rates
5. **Regularization:** Depth loss, TV loss, masking losses

**Rendering (Holoscan Pipeline):**
1. **EndoNeRFLoaderOp:** Loads frame data (pose, time)
2. **GsplatLoaderOp:** Loads Gaussians + deformation network
3. **GsplatRenderOp:** Applies deformation(time) → rasterizes
4. **HolovizOp:** Real-time visualization

### Temporal Deformation Network

- **Architecture:** HexPlane 4D grid + MLP
- **Input:** 3D position + time value [0, 1]
- **Output:** Deformed position, scale, rotation, opacity
- **Inference:** Direct PyTorch (no conversion, full precision)

### Gaussian Splatting

- **Renderer:** gsplat library (CUDA-accelerated)
- **Splats:** ~50,000 gaussians (full training)
- **SH Degree:** 3 (16 coefficients per gaussian)
- **Resolution:** 640×512 pixels
- **Format:** RGB (3 channels)

### Performance

**Tested Configuration:**
- **GPU:** NVIDIA RTX 6000 Ada Generation
- **Container:** Holoscan SDK 3.7.0
- **Training Time:**
  - Quick: ~10 minutes (30 frames, 500 iter)
  - Full: ~30 minutes (63 frames, 2000 iter)
- **Rendering:** Real-time >30 FPS

**Quality Metrics (training_full):**
- **PSNR:** ~36-38 dB
- **SSIM:** ~0.80
- **Gaussians:** ~50,000 splats
- **Deformation:** Smooth temporal consistency

---

## File Structure

```
surgical_scene_recon/
├── README.md                       # This file
├── Dockerfile                      # Production container
├── metadata.json                   # Holohub CLI configuration
│
├── demo_dynamic_rendering_viz.py   # Dynamic rendering demo
├── train_standalone.py             # Training script
├── run_surgical_recon.py           # Unified Python interface
├── run_complete_workflow.py        # Python workflow orchestrator
│
├── operators/                      # Custom Holoscan operators
│   ├── endonerf_loader_op.py      # Data loading
│   ├── gsplat_loader_op.py        # Checkpoint loading
│   ├── gsplat_render_op.py        # Rendering
│   └── image_saver_op.py          # Image saving
│
├── training/                       # Training code
│   ├── gsplat_train.py            # Main training script
│   ├── scene/                     # Data loaders
│   └── utils/                     # Training utilities
│
└── tests/                          # Unit tests
    ├── test_dynamic_rendering_viz.py
    ├── test_static_rendering_viz.py
    ├── test_data_loading.py
    └── test_minimal_holoviz.py
```

---

## Citation

If you use this work, please cite:

**EndoNeRF:**
```bibtex
@inproceedings{wang2022endonerf,
  title={EndoNeRF: Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery},
  author={Wang, Yuehao and Yifan, Wang and Tao, Rui and others},
  booktitle={MICCAI},
  year={2022}
}
```

**3D Gaussian Splatting:**
```bibtex
@article{kerbl20233d,
  title={3d gaussian splatting for real-time radiance field rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  year={2023}
}
```

**gsplat Library:**
```bibtex
@software{ye2024gsplat,
  title={gsplat},
  author={Ye, Vickie and Turkulainen, Matias and others},
  year={2024},
  url={https://github.com/nerfstudio-project/gsplat}
}
```

---

## License

This application is licensed under Apache 2.0. See individual files for specific licensing:
- Application code: Apache 2.0 (NVIDIA)
- Training utilities: MIT License (EndoGaussian Project)
- Spherical harmonics utils: BSD-2-Clause (PlenOctree)

---

## Support

For issues or questions:
- Open an issue on the [HoloHub repository](https://github.com/nvidia-holoscan/holohub)
- Refer to the [Holoscan SDK documentation](https://docs.nvidia.com/holoscan/)

