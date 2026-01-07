# Surgical Scene Reconstruction with Gaussian Splatting

This application demonstrates real-time 3D surgical scene reconstruction by combining **Holoscan SDK** for high-performance streaming, **3D Gaussian Splatting** for neural 3D representation, and **temporal deformation networks** for accurate modeling of dynamic tissue.

Real-time 3D surgical scene reconstruction using Gaussian Splatting in a Holoscan streaming pipeline with temporal deformation for accurate tissue modeling.

![Training Visualization - Ground Truth vs Rendered](train_gt_animation.gif)


The application provides a complete end-to-end pipeline—from raw surgical video to real-time 3D reconstruction. Researchers and developers can use it to train custom models on their own endoscopic data and visualize results with GPU-accelerated rendering.

Features of this application include:

- **Real-time Visualization:** Stream surgical scene reconstruction at >30 FPS using Holoscan
- **Temporal Deformation:** Accurate per-frame tissue modeling as it deforms over time
- **Tool Removal:** Tissue-only reconstruction mode (surgical instruments automatically excluded)
- **End-to-End Training:** On-the-fly model training from streamed endoscopic data
- **Two Operation Modes:** Inference-only (with pre-trained checkpoint) OR train-then-render
- **Production Ready:** Tested and optimized Holoscan pipeline with complete Docker containerization

It takes input from EndoNeRF surgical datasets (RGB images + stereo depth + camera poses + tool masks). It processes the input using multi-frame Gaussian Splatting with a 4D spatiotemporal deformation network. And it outputs real-time 3D tissue reconstruction without surgical instruments.

It is ideal for uses cases, such as:

- Surgical scene understanding and visualization
- Tool-free tissue reconstruction for analysis
- Research in surgical vision and 3D reconstruction
- Development of real-time surgical guidance systems

## Quick Start

### Step 1: Clone the HoloHub Repository

```bash
git clone https://github.com/nvidia-holoscan/holohub.git
cd holohub
```

### Step 2: Read and Agree to the Terms and Conditions of the EndoNeRF Sample Dataset

1. Read and agree to the [Terms and Conditions](https://docs.google.com/document/d/1P6q2hXoGpVMKeD-PpjYYdZ0Yx1rKZdJF1rXxpobbFMY/edit?usp=share_link) for the EndoNeRF dataset.
1. EndoNeRF sample dataset is being downloaded automatically when building the application. 
1. Optionally, for manual download of the dataset, refer to the [Data](#pulling-soft-tissues-dataset) section below.
1. Optionally, if you do not agree to the terms and conditions, set the `HOLOHUB_DOWNLOAD_DATASETS` environment variable to `OFF` and manually download the dataset and place it in the correct location by following the instructions in the [Data](#pulling-soft-tissues-dataset) section below.

    ```bash
    export HOLOHUB_DOWNLOAD_DATASETS=OFF
    ```

### Step 3: Run Training

To run the virtual surgical training:

  ```bash
  ./holohub run surgical_scene_recon train
  ```

### Step 4: Dynamic Rendering with a Trained Model

After training completes, to visualize your results in real-time, run the surgical render:

```bash
./holohub run surgical_scene_recon render
```

![Dynamic Rendering Visualization](surg_recon_inference.gif)

## Pulling Soft Tissues Dataset

This application uses the **EndoNeRF "pulling_soft_tissues" dataset**, which contains:

- 63 RGB endoscopy frames (640×512 pixels)
- Corresponding depth maps
- Tool segmentation masks for instrument removal
- Camera poses and bounds (poses_bounds.npy)

### Download the Dataset

You can download the dataset from one of the following locations:

* 📦 Direct Google Drive: <https://drive.google.com/drive/folders/1zTcX80c1yrbntY9c6-EK2W2UVESVEug8?usp=sharing>

  1. In the Google Drive folder, you'll see:

      - `cutting_tissues_twice`
      - `pulling_soft_tissues`
  
  1. Download `pulling_soft_tissues`.

* Visit the [EndoNeRF repository](https://github.com/med-air/EndoNeRF).

### Dataset Setup

The dataset will be automatically used by the application when placed in the correct location. Refer to the [HoloHub glossary](../../README.md#Glossary) for definitions of HoloHub-specific directory terms used below.

To place the dataset at `<HOLOHUB_ROOT>/data/EndoNeRF/pulling/`:

1. From the HoloHub root directory:
    ```bash
    mkdir -p data/EndoNeRF
    ```

1. Extract and move (or copy) the downloaded dataset: 

    ```bash
    mv /path/to/pulling_soft_tissues data/EndoNeRF/pulling
    ```

**Important:** The dataset MUST be physically at the path above, do NOT use symlinks. Docker containers cannot follow symlinks outside mounted volumes.

### Verify the Dataset Structure

Verify that your dataset has this structure:

```text
<HOLOHUB_ROOT>/
└── data/
    └── EndoNeRF/
        └── pulling/
            ├── images/              # 63 RGB frames (.png)
            ├── depth/               # 63 depth maps (.png)
            ├── masks/               # 63 tool masks (.png)
            └── poses_bounds.npy     # Camera poses (8.5 KB)
```

## Model

The application uses **3D Gaussian Splatting** with a **temporal deformation network** for dynamic scene reconstruction:

### Gaussian Splatting

- **Architecture:** 3D Gaussians with learned position, scale, rotation, opacity, and color
- **Initialization:** Multi-frame point cloud (~30,000-50,000 points from all frames)
- **Renderer:** `gsplat` library (CUDA-accelerated differentiable rasterization)
- **Spherical Harmonics:** Degree 3 (16 coefficients per gaussian for view-dependent color)
- **Resolution:** 640×512 pixels (RGB, three channels)

### Temporal Deformation Network

- **Architecture:** HexPlane 4D spatiotemporal grid + MLP decoder
- **Input:** 3D position + normalized time value [0, 1]
- **Output:** Deformed position, scale, rotation, and opacity changes
- **Training:** Two-stage process (coarse: static, fine: with deformation)
- **Inference:** Direct PyTorch (no conversion, full precision)

### Training Process

The application trains in two stages:

1. **Coarse Stage:** Learn base static Gaussians without deformation
2. **Fine Stage:** Add temporal deformation network for dynamic tissue modeling

The training uses:

- **Multi-modal Data:** RGB images, depth maps, tool segmentation masks
- **Loss Functions:** RGB loss, depth loss, TV loss, masking losses
- **Optimization:** Adam optimizer with batch-size scaled learning rates
- **Tool Removal:** Segmentation masks applied during training for tissue-only reconstruction

The default training command trains a model on all 63 frames with 2000 iterations, producing smooth temporal deformation and high-quality reconstruction.

Training outputs are saved to `<HOLOHUB_APP_BIN>/output/trained_model/`, where `<HOLOHUB_APP_BIN>` is `<HOLOHUB_ROOT>/build/surgical_scene_recon/applications/surgical_scene_recon/` by default.

- `ckpts/fine_best_psnr.pt` - Best checkpoint (use for rendering)
- `ckpts/fine_step00XXX.pt` - Regular checkpoints
- `logs/` - Training logs
- `tb/` - TensorBoard logs
- `renders/` - Saved render frames

You can view training logs using TensorBoard:

```bash
tensorboard --logdir <HOLOHUB_APP_BIN>/output/trained_model/tb
```

## Holoscan Pipeline Architecture

The real-time rendering uses the following Holoscan operators:

```text
EndoNeRFLoaderOp → GsplatLoaderOp → GsplatRenderOp → HolovizOp
                                                    ↓
                                              ImageSaverOp
```

**Components:**

- **EndoNeRFLoaderOp:** Streams camera poses and timestamps
- **GsplatLoaderOp:** Loads checkpoint and deformation network
- **GsplatRenderOp:** Applies temporal deformation and renders
- **HolovizOp:** Real-time GPU-accelerated visualization
- **ImageSaverOp:** Optional frame saving

## Requirements

- **Hardware:**
  - NVIDIA GPU (RTX 3000+ series recommended, tested on RTX 6000 Ada Generation)
  - ~2 GB free disk space (dataset)
  - ~30 GB free disk space (Docker container)
- **Software:**
  - Docker with NVIDIA GPU support
  - X11 display server (for visualization)
  - Holoscan SDK 3.7.0 or later (automatically provided in container)

## Testing

We provide integration tests that can be run with the following command to test the application for training and inference:

```bash
./holohub test surgical_scene_recon --verbose
```

## Technical Details

### Training Pipeline (gsplat_train.py)

1. **Data Loading:** EndoNeRF parser loads RGB, depth, masks, poses
2. **Initialization:** Multi-frame point cloud (~30k points)
3. **Two-Stage Training:**
   - **Coarse:** Learn base Gaussians (no deformation)
   - **Fine:** Add temporal deformation network
4. **Optimization:** Adam with batch-size scaled learning rates
5. **Regularization:** Depth loss, TV loss, masking losses

### Performance

**Tested Configuration:**

- **GPU:** NVIDIA RTX 6000 Ada Generation
- **Container:** Holoscan SDK 3.7.0
- **Training Time:** ~5 minutes (63 frames, 2000 iterations)
- **Rendering:** Real-time >30 FPS

**Quality Metrics (train mode):**

- **PSNR:** ~36-38 dB
- **SSIM:** ~0.80
- **Gaussians:** ~50,000 splats
- **Deformation:** Smooth temporal consistency

## Troubleshooting

### Problem: "FileNotFoundError: poses_bounds.npy not found"

- **Cause:** Dataset not in correct location or symlink used
- **Solution:** Ensure dataset is physically at `<HOLOHUB_ROOT>/data/EndoNeRF/pulling/`
- **Verify:** Run `file data/EndoNeRF` - should show "directory", not "symbolic link"

### Problem: "Unable to find image holohub-surgical_scene_recon"

- **Cause:** Container not built yet
- **Solution:** Remove `--no-docker-build` flag (let CLI build automatically)
- **Or:** Manually build: `./holohub build-container surgical_scene_recon`

### Problem: Holoviz window doesn't appear

- **Cause:** X11 forwarding not enabled
- **Solution:** Run `xhost +local:docker` before training
- **Verify:** Check `echo $DISPLAY` shows a value

### Problem: GPU out of memory

- **Cause:** Another process using GPU
- **Solution:** Check `nvidia-smi` and stop other processes
- **Or:** Reduce batch size (advanced - edit training config)

## Acknowledgements

### Citation

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

### License

This application is licensed under Apache 2.0. See individual files for specific licensing:

- Application code: Apache 2.0 (NVIDIA)
- Training utilities: MIT License (EndoGaussian Project)
- Spherical harmonics utils: BSD-2-Clause (PlenOctree)
