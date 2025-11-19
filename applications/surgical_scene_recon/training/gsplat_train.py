# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
EndoGaussian Trainer with gsplat Backend
========================================

Modern implementation combining:
- gsplat's rasterization and optimization strategies
- EndoGaussian's medical imaging features (depth loss, masks, deformation)
- Two-stage training (coarse → fine)

Architecture:
    EndoConfig: Configuration dataclass
    EndoRunner: Main training class (inherits patterns from gsplat)
    EndoNeRFParser: Data loader for EndoNeRF/SCARED datasets
    EndoNeRFDataset: PyTorch dataset wrapper
"""

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import tqdm
import yaml
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.regression import pearson_corrcoef
from typing_extensions import Literal

# gsplat imports
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam

# Local imports
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss
from utils.image_utils import psnr


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EndoConfig:
    """Configuration for EndoGaussian with gsplat backend"""
    
    # ========== Data Configuration ==========
    data_dir: str = "data/EndoNeRF/pulling"
    data_factor: int = 1  # Downsample factor
    result_dir: str = "results/endo_gsplat"
    test_every: int = 8
    
    # Data type
    dataset_type: Literal["endonerf", "scared"] = "endonerf"
    depth_mode: Literal["binocular", "monocular"] = "binocular"  # EndoNeRF default
    
    # ========== Training Configuration ==========
    max_steps: int = 1_500
    batch_size: int = 1
    steps_scaler: float = 1.0
    
    # Two-stage training (EndoGaussian specific)
    two_stage_training: bool = True
    coarse_iterations: int = 300
    fine_iterations: int = 1_500  # max_steps
    
    # ========== Gaussian Initialization ==========
    init_type: str = "sfm"  # or "random"
    init_num_pts: int = 30_000
    init_extent: float = 3.0
    init_opa: float = 0.1
    init_scale: float = 1.0
    sh_degree: int = 3
    sh_degree_interval: int = 500
    
    # ========== Optimization ==========
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    
    # ========== Loss Weights ==========
    ssim_lambda: float = 0.2
    lpips_lambda: float = 0.0
    
    # Endoscopy-specific losses
    depth_lambda: float = 0.001  # Nearly RGB-only, minimal depth constraint
    pearson_lambda: float = 0.001  # For monocular depth
    tv_lambda: float = 0.03  # TV loss weight
    
    # Priority 2: Accurate Mask + Targeted TV Loss
    accurate_mask: bool = True  # Create invisible mask from all frames
    tv_on_occluded_only: bool = True  # Apply TV loss only to occluded regions
    
    # Priority 1: Multi-Frame Point Cloud Initialization (MOST CRITICAL!)
    multiframe_init: bool = True  # Accumulate depth/color across all frames
    multiframe_sample_rate: int = 3  # Sample every Nth pixel for efficiency
    
    # ========== Masking Mode ==========
    use_masks: bool = True  # Master switch: Enable/disable all masking
    # When use_masks=False: Full scene reconstruction (tissue + tools)
    # When use_masks=True: Tissue-only reconstruction (tools removed)
    
    # ========== Deformation Network ==========
    use_deformation: bool = True
    deformation_lr: float = 1.0e-5  # Further reduced for stability
    grid_lr: float = 1.0e-5  # Further reduced for stability
    
    # Deformation regularization (increased from 0.0 for stability)
    time_smoothness_weight: float = 0.01  # Temporal smoothness
    l1_time_planes_weight: float = 0.01  # L1 regularization on temporal grids
    plane_tv_weight: float = 0.01  # Total variation on spatial grids
    
    # Tool masking for tissue reconstruction
    use_mask: bool = True  # Use masks in loss computation (exclude tool pixels)
    enable_tool_masking: bool = False  # Update deformation table based on tool masks
    tool_mask_threshold: int = 1  # Min frames in tool region to mark as non-deformable
    tool_mask_update_interval: int = 500  # Update deformation table every N steps
    
    # Depth supervision in tool regions (NEW)
    depth_supervise_tools: bool = False  # Use depth supervision even in tool regions
    # When True: RGB masked (exclude tools), Depth unmasked (include tools)
    # When False: Both RGB and depth masked (current behavior)
    
    # Deformation network architecture
    bounds: float = 1.5
    kplanes_config: Dict = field(default_factory=lambda: {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 64,
        'resolution': [64, 64, 64, 100]
    })
    multires: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    defor_depth: int = 0
    net_width: int = 32
    timebase_pe: int = 6
    posebase_pe: int = 10
    scale_rotation_pe: int = 10
    opacity_pe: int = 10
    timenet_width: int = 64
    timenet_output: int = 32
    no_grid: bool = False
    no_dx: bool = False
    no_ds: bool = False
    no_dr: bool = False
    no_do: bool = False
    
    # ========== Densification Strategy ==========
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=lambda: DefaultStrategy(verbose=True)
    )
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False
    
    # Stage-specific thresholds
    opacity_threshold_coarse: float = 0.05
    opacity_threshold_fine_init: float = 0.05
    opacity_threshold_fine_after: float = 0.005
    densify_grad_threshold_coarse: float = 0.0002
    densify_grad_threshold_fine_init: float = 0.0002
    densify_grad_threshold_after: float = 0.0002
    
    # ========== Rendering ==========
    near_plane: float = 0.01
    far_plane: float = 1e10
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    random_bkgd: bool = False
    
    # ========== Logging & Saving ==========
    tb_every: int = 100
    tb_save_image: bool = False
    eval_steps: List[int] = field(default_factory=lambda: [1_200, 1_500])  # Eval at end of fine stage
    save_steps: List[int] = field(default_factory=lambda: [1_200, 1_500])  # Save at end of fine stage
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [1_200, 1_500])
    disable_video: bool = False
    
    # ========== Viewer ==========
    disable_viewer: bool = True  # Disable by default for endoscopy
    port: int = 8080
    
    # ========== Advanced ==========
    global_scale: float = 1.0
    normalize_world_space: bool = True
    lpips_net: Literal["vgg", "alex"] = "vgg"
    
    def adjust_steps(self, factor: float):
        """Scale training steps by a factor"""
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.coarse_iterations = int(self.coarse_iterations * factor)
        self.fine_iterations = int(self.fine_iterations * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        
        # Adjust strategy intervals
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.refine_start_iter = int(self.strategy.refine_start_iter * factor)
            self.strategy.refine_stop_iter = int(self.strategy.refine_stop_iter * factor)
            self.strategy.reset_every = int(self.strategy.reset_every * factor)
            self.strategy.refine_every = int(self.strategy.refine_every * factor)
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.refine_start_iter = int(self.strategy.refine_start_iter * factor)
            self.strategy.refine_stop_iter = int(self.strategy.refine_stop_iter * factor)
            self.strategy.refine_every = int(self.strategy.refine_every * factor)


# ============================================================================
# EndoNeRF Data Loading (PHASE 2 - IMPLEMENTED)
# ============================================================================

class EndoNeRFParser:
    """Parser for EndoNeRF dataset format
    
    Wraps the existing EndoNeRF_Dataset to provide gsplat-compatible interface.
    Provides:
    - points: Initial point cloud positions [N, 3]
    - points_rgb: Initial point cloud colors [N, 3]
    - camtoworlds: Camera poses [N, 4, 4]
    - Ks_dict: Camera intrinsics dict
    - imsize_dict: Image sizes dict
    - scene_scale: Scene normalization scale
    """
    
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        test_every: int = 8,
        depth_mode: str = "monocular",
        dataset_type: str = "endonerf",
        multiframe_init: bool = True,
        multiframe_sample_rate: int = 3,
        use_masks: bool = True,
    ):
        """Initialize EndoNeRF parser
        
        Args:
            data_dir: Path to dataset directory
            factor: Downsample factor for images
            test_every: Every Nth frame is test frame
            depth_mode: 'monocular' or 'binocular'
            dataset_type: 'endonerf' or 'scared'
            multiframe_init: Use multi-frame accumulation (Priority 1)
            multiframe_sample_rate: Sample every Nth pixel
            use_masks: If True, exclude tools; if False, include everything
        """
        self.data_dir = data_dir
        self.factor = factor
        self.test_every = test_every
        self.depth_mode = depth_mode
        self.dataset_type = dataset_type
        self.multiframe_init = multiframe_init
        self.multiframe_sample_rate = multiframe_sample_rate
        self.use_masks = use_masks
        
        # Load using existing EndoNeRF loaders
        from scene.endo_loader import EndoNeRF_Dataset, SCARED_Dataset
        
        if dataset_type == "endonerf":
            self.endo_dataset = EndoNeRF_Dataset(
                datadir=data_dir,
                downsample=factor,
                test_every=test_every,
                mode=depth_mode
            )
        elif dataset_type == "scared":
            self.endo_dataset = SCARED_Dataset(
                datadir=data_dir,
                downsample=factor,
                test_every=test_every,
                mode=depth_mode
            )
        else:
            raise ValueError(f"Dataset type {dataset_type} not supported in Phase 2")
        
        # Get initial point cloud
        if self.multiframe_init:
            # Priority 1: Multi-frame accumulation (SurgicalGaussian approach)
            print("[Priority 1] Using multi-frame point cloud initialization...")
            pts, colors = accumulate_multiframe_pointcloud(
                self.endo_dataset,
                depth_mode=self.depth_mode,
                sample_rate=self.multiframe_sample_rate,
                use_masks=self.use_masks,  # Pass masking mode to initialization
            )
            self.points = pts.astype(np.float32)
            self.points_rgb = (colors * 255.0).astype(np.uint8)  # gsplat expects 0-255
        else:
            # Original: Single/random frame approach
            print(f"[Phase 2] Loading initial point cloud from {dataset_type} dataset...")
            pts, colors, normals = self.endo_dataset.get_init_pts()
            self.points = pts.astype(np.float32)
            self.points_rgb = (colors * 255.0).astype(np.uint8)  # gsplat expects 0-255
        
        print(f"[Phase 2] Loaded {self.points.shape[0]} initial points")
        
        # Extract camera parameters
        self._extract_camera_params()
        
        # Compute scene scale (from point cloud extent)
        point_extent = np.max(np.linalg.norm(self.points, axis=1))
        self.scene_scale = max(point_extent, 1.0)
        
        print(f"[Phase 2] Scene scale: {self.scene_scale:.3f}")
    
    def _extract_camera_params(self):
        """Extract camera poses and intrinsics from EndoNeRF dataset"""
        # Get camera info from EndoNeRF dataset
        n_frames = len(self.endo_dataset.image_paths)
        
        self.camtoworlds = []
        self.Ks_dict = {}
        self.imsize_dict = {}
        
        for idx in range(n_frames):
            R, T = self.endo_dataset.image_poses[idx]
            
            # Convert R, T to c2w (camera-to-world)
            # EndoNeRF stores w2c components, need to convert
            R_t = np.transpose(R)  # Transpose back
            w2c = np.eye(4)
            w2c[:3, :3] = R_t
            w2c[:3, 3] = T
            c2w = np.linalg.inv(w2c)
            
            self.camtoworlds.append(c2w)
            
            # Store intrinsics (same for all frames in EndoNeRF)
            if idx == 0:
                K = self.endo_dataset.K
                self.Ks_dict[idx] = K
                self.imsize_dict[idx] = self.endo_dataset.img_wh
        
        self.camtoworlds = np.stack(self.camtoworlds, axis=0).astype(np.float32)
        
        # For gsplat compatibility, replicate K for all frames
        for idx in range(1, n_frames):
            self.Ks_dict[idx] = self.Ks_dict[0]
            self.imsize_dict[idx] = self.imsize_dict[0]


class EndoNeRFDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for EndoNeRF
    
    Provides data in gsplat-compatible format:
    - image: [H, W, 3] tensor (RGB, 0-1 range)
    - depth: [H, W] tensor
    - mask: [H, W] bool tensor
    - camtoworld: [4, 4] tensor
    - K: [3, 3] tensor
    - image_id: int
    - time: float (for deformation, normalized 0-1)
    """
    
    def __init__(
        self,
        parser: EndoNeRFParser,
        split: str = "train",
        load_depths: bool = True,
    ):
        """Initialize dataset
        
        Args:
            parser: EndoNeRFParser instance
            split: 'train', 'test', or 'video'
            load_depths: Whether to load depth maps
        """
        self.parser = parser
        self.split = split
        self.load_depths = load_depths
        
        # Get cameras from EndoNeRF dataset
        endo_dataset = parser.endo_dataset
        
        # Get the appropriate indices
        if split == "train":
            self.indices = endo_dataset.train_idxs
        elif split == "test":
            self.indices = endo_dataset.test_idxs
        elif split == "video":
            self.indices = endo_dataset.video_idxs
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"[Phase 2] {split} dataset: {len(self.indices)} frames")
        
        # Cache for loaded data (optional optimization)
        self._cache = {}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single data sample
        
        Returns:
            dict with keys: image, depth, mask, camtoworld, K, image_id, time
        """
        # Get actual frame index
        frame_idx = self.indices[idx]
        
        # Check cache
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        
        endo_dataset = self.parser.endo_dataset
        
        # Load image
        from PIL import Image
        
        image_path = endo_dataset.image_paths[frame_idx]
        image = Image.open(image_path)
        image = np.array(image) / 255.0  # [H, W, 3], 0-1 range
        image = torch.from_numpy(image).float()
        
        # Load depth
        depth_path = endo_dataset.depth_paths[frame_idx]
        if self.parser.depth_mode == 'binocular':
            depth = np.array(Image.open(depth_path))
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth = np.clip(depth, close_depth, inf_depth)
        elif self.parser.depth_mode == 'monocular':
            depth = np.array(Image.open(depth_path))[..., 0] / 255.0
            depth[depth!=0] = (1 / depth[depth!=0]) * 0.4
            depth[depth==0] = depth.max()
        depth = torch.from_numpy(depth).float()
        if depth.ndim == 2:
            depth = depth.unsqueeze(-1)  # [H, W, 1]
        depth = depth.squeeze(-1)  # [H, W]
        
        # Load mask
        mask_path = endo_dataset.masks_paths[frame_idx]
        mask = Image.open(mask_path)
        mask = 1 - np.array(mask) / 255.0  # Invert: 1 for valid, 0 for tool
        mask = torch.from_numpy(mask).bool()
        if mask.ndim == 3:
            mask = mask[..., 0]  # [H, W]
        
        # Get camera pose (c2w)
        camtoworld = torch.from_numpy(self.parser.camtoworlds[frame_idx]).float()
        
        # Get intrinsics
        K = torch.from_numpy(self.parser.Ks_dict[0]).float()
        
        # Get time (normalized 0-1)
        time = endo_dataset.image_times[frame_idx]
        
        data = {
            'image': image,  # [H, W, 3]
            'depth': depth,  # [H, W]
            'mask': mask,    # [H, W]
            'camtoworld': camtoworld,  # [4, 4]
            'K': K,          # [3, 3]
            'image_id': frame_idx,
            'time': torch.tensor(time, dtype=torch.float32),
        }
        
        return data


# ============================================================================
# Main Trainer Class
# ============================================================================

class EndoRunner:
    """Main training runner for EndoGaussian with gsplat backend
    
    This class integrates:
    1. gsplat's modern rasterization and optimization
    2. EndoGaussian's medical imaging features
    3. Two-stage training (coarse → fine)
    4. Deformation network for dynamic scenes
    """
    
    def __init__(
        self,
        local_rank: int,
        world_rank: int,
        world_size: int,
        cfg: EndoConfig,
    ) -> None:
        """Initialize the EndoRunner
        
        Args:
            local_rank: Local GPU rank
            world_rank: Global rank (for distributed training)
            world_size: Total number of processes
            cfg: Configuration object
        """
        torch.manual_seed(42 + local_rank)
        
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        
        # Setup output directories
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        
        # Phase 2: Load actual data
        print(f"[Phase 2] Loading {cfg.dataset_type} dataset from {cfg.data_dir}...")
        self.parser = EndoNeRFParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            test_every=cfg.test_every,
            depth_mode=cfg.depth_mode,
            dataset_type=cfg.dataset_type,
            multiframe_init=cfg.multiframe_init,
            multiframe_sample_rate=cfg.multiframe_sample_rate,
            use_masks=cfg.use_masks,  # Pass masking mode to parser
        )
        
        # Phase 2: Create actual datasets
        self.trainset = EndoNeRFDataset(self.parser, split="train")
        self.valset = EndoNeRFDataset(self.parser, split="test")
        self.scene_scale = self.parser.scene_scale * cfg.global_scale
        
        # Phase 3: Initialize Gaussian splats
        self.splats, self.optimizers = create_splats_with_optimizers(
            parser=self.parser,
            cfg=cfg,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print(f"[Phase 3] Model initialized with {len(self.splats['means'])} Gaussians")
        
        # Phase 3: Initialize densification strategy
        cfg.strategy.check_sanity(self.splats, self.optimizers)
        
        if isinstance(cfg.strategy, DefaultStrategy):
            self.strategy_state = cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(cfg.strategy, MCMCStrategy):
            self.strategy_state = cfg.strategy.initialize_state()
        else:
            raise ValueError(f"Unknown strategy: {type(cfg.strategy)}")
        
        # Phase 6: Initialize deformation network
        if cfg.use_deformation:
            print("[Phase 6] Initializing deformation network for dynamic scenes...")
            from scene.deformation import deform_network
            
            # Create args object for deformation network
            from argparse import Namespace
            deform_args = Namespace(
                bounds=cfg.bounds,
                kplanes_config=cfg.kplanes_config,
                multires=cfg.multires,
                no_grid=cfg.no_grid,
                no_dx=cfg.no_dx,
                no_ds=cfg.no_ds,
                no_dr=cfg.no_dr,
                no_do=cfg.no_do,
                net_width=cfg.net_width,
                timebase_pe=cfg.timebase_pe,
                defor_depth=cfg.defor_depth,
                posebase_pe=cfg.posebase_pe,
                scale_rotation_pe=cfg.scale_rotation_pe,
                opacity_pe=cfg.opacity_pe,
                timenet_width=cfg.timenet_width,
                timenet_output=cfg.timenet_output,
            )
            
            self.deform_net = deform_network(deform_args).to(self.device)
            
            # Initialize deformation table (which Gaussians can be deformed)
            # Start with all Gaussians marked as deformable
            N = len(self.splats["means"])
            self._deformation_table = torch.ones(N, dtype=torch.bool, device=self.device)
            self._deformation_accum = torch.zeros((N, 3), device=self.device)
            
            # Create separate optimizers for MLP and grid
            self.deform_optimizers = [
                torch.optim.Adam(
                    self.deform_net.get_mlp_parameters(),
                    lr=cfg.deformation_lr * self.scene_scale,
                ),
                torch.optim.Adam(
                    self.deform_net.get_grid_parameters(),
                    lr=cfg.grid_lr * self.scene_scale,
                ),
            ]
            
            print("[Phase 6] Deformation network initialized")
            print(f"[Phase 6]   MLP parameters: {sum(p.numel() for p in self.deform_net.get_mlp_parameters())}")
            print(f"[Phase 6]   Grid parameters: {sum(p.numel() for p in self.deform_net.get_grid_parameters())}")
            print(f"[Phase 6]   Deformable Gaussians: {self._deformation_table.sum().item()}/{N}")
        else:
            self.deform_net = None
            self.deform_optimizers = []
        
        # TODO Phase 3: Initialize metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=cfg.lpips_net, normalize=(cfg.lpips_net == "alex")
        ).to(self.device)
        
        # Track best PSNR for checkpoint saving
        self.best_psnr = -float('inf')
        self.best_psnr_step = -1
        
        # Priority 2: Create invisible mask if accurate_mask AND use_masks are enabled
        if cfg.use_masks and cfg.accurate_mask:
            print("\n[Priority 2] Creating invisible mask from all frames...")
            self.invisible_mask = self._create_invisible_mask()
            coverage = (self.invisible_mask > 0).sum().item() / self.invisible_mask.numel() * 100
            print(f"[Priority 2] Invisible regions: {coverage:.1f}% of image")
            print(f"[Priority 2] Mask shape: {self.invisible_mask.shape}")
        else:
            self.invisible_mask = None
            if not cfg.use_masks:
                print("[Priority 2] Masking disabled - Full scene reconstruction mode")
            else:
                print("[Priority 2] Accurate mask disabled")
        
        print(f"\n[Phase 1] EndoRunner initialized on {self.device}")
        print(f"[Phase 1] Configuration: two_stage={cfg.two_stage_training}, "
              f"depth_mode={cfg.depth_mode}, use_deformation={cfg.use_deformation}")
        print(f"[Priority 2] Accurate mask: {cfg.accurate_mask}, TV on occluded only: {cfg.tv_on_occluded_only}")
        print(f"[Phase 1] Masking: use_mask={cfg.use_mask}, enable_tool_masking={cfg.enable_tool_masking}, "
              f"depth_supervise_tools={cfg.depth_supervise_tools}")
    
    def _create_invisible_mask(self) -> Tensor:
        """
        Create invisible mask from all tool masks across all frames.
        
        Priority 2 implementation: Identifies regions that are chronically occluded
        by surgical tools across the video sequence.
        
        Returns:
            invisible_mask: [H, W] tensor where 1=invisible (always occluded), 0=visible
        """
        # Get mask paths from dataset
        endo_dataset = self.parser.endo_dataset
        mask_paths = endo_dataset.masks_paths
        
        # Create union of all tool masks
        invisible_mask = create_invisible_mask_from_paths(mask_paths, device=self.device)
        
        # Dilate for safety margin (following SurgicalGaussian approach)
        invisible_mask = dilate_invisible_mask(invisible_mask, kernel_size=5, iterations=2)
        
        return invisible_mask
    
    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        time_idx: Optional[Tensor] = None,
        stage: str = "fine",  # Phase 6: Stage determines if deformation is applied
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Rasterize Gaussian splats with stage-dependent deformation
        
        Phase 4: Implemented using gsplat.rasterization()
        Phase 6: Fixed deformation - only applies in fine stage to subset
        Phase 8: CRITICAL FIX - Removed rendered output masking (was preventing tissue reconstruction)
        
        Args:
            camtoworlds: Camera poses [B, 4, 4]
            Ks: Camera intrinsics [B, 3, 3]
            width: Image width
            height: Image height
            time_idx: Time indices for deformation [B] (Phase 6)
            stage: 'coarse' or 'fine' - deformation only in fine!
            rasterize_mode: "classic" or "antialiased"
            camera_model: "pinhole", "ortho", or "fisheye"
            **kwargs: Additional rendering options
            
        Returns:
            render_colors: Rendered images [B, H, W, C] - UNMASKED (masks used in loss only)
            render_alphas: Alpha values [B, H, W, 1]
            info: Additional rendering info dict
        """
        # Get canonical Gaussian parameters (RAW, before activation)
        means_canonical = self.splats["means"]  # [N, 3]
        scales_raw = self.splats["scales"]  # [N, 3] in log space
        quats_raw = self.splats["quats"]  # [N, 4]
        opacities_raw = self.splats["opacities"]  # [N,] in logit space
        
        # Phase 6: Apply deformation ONLY in fine stage
        # Debug: Check deformation conditions (only in fine stage)
        if stage == "fine" and not hasattr(self, '_deform_cond_logged'):
            print("\n[DEBUG] Deformation conditions (fine stage):")
            print(f"  stage={stage}")
            print(f"  time_idx={time_idx}")
            print(f"  time_idx is not None: {time_idx is not None}")
            print(f"  hasattr deform_net: {hasattr(self, 'deform_net')}")
            print(f"  deform_net is not None: {hasattr(self, 'deform_net') and self.deform_net is not None}")
            self._deform_cond_logged = True
        
        if (stage == "fine" and time_idx is not None and 
            hasattr(self, 'deform_net') and self.deform_net is not None):
            
            deform_mask = self._deformation_table
            N = means_canonical.shape[0]
            
            # Expand time to match number of Gaussians [N, 1]
            # time_idx comes as [B] or [B, 1], need to expand to [N, 1] for all Gaussians
            if time_idx.dim() == 0:
                # Scalar
                time_expanded = time_idx.view(1, 1).repeat(N, 1)
            elif time_idx.dim() == 1:
                # [B] -> [1, 1] -> [N, 1]
                time_expanded = time_idx[0].view(1, 1).repeat(N, 1)
            else:
                # [B, 1] -> [1, 1] -> [N, 1]
                time_expanded = time_idx[0].view(1, 1).repeat(N, 1)
            
            # Apply deformation to SUBSET marked by mask  
            # Note: opacity needs to be [N, 1] for deformation network
            means_deformed, scales_deformed, quats_deformed, opacities_deformed = self.deform_net(
                point=means_canonical[deform_mask],
                scales=scales_raw[deform_mask],
                rotations=quats_raw[deform_mask],
                opacity=opacities_raw[deform_mask].unsqueeze(-1),  # [N_masked, 1]
                times_sel=time_expanded[deform_mask],
            )
            
            # Debug: Check deformation output shapes and values
            if not hasattr(self, '_deform_debug_logged'):
                print("\n[DEBUG] Deformation output shapes (first call):")
                print(f"  means_deformed: {means_deformed.shape}, range: [{means_deformed.min():.3f}, {means_deformed.max():.3f}]")
                print(f"  scales_deformed: {scales_deformed.shape}, range: [{scales_deformed.min():.3f}, {scales_deformed.max():.3f}]")
                print(f"  quats_deformed: {quats_deformed.shape}, range: [{quats_deformed.min():.3f}, {quats_deformed.max():.3f}]")
                print(f"  opacities_deformed: {opacities_deformed.shape}, range: [{opacities_deformed.min():.3f}, {opacities_deformed.max():.3f}]")
                print(f"  time_expanded shape: {time_expanded.shape}, sample: {time_expanded[0]}")
                print(f"  deform_mask sum: {deform_mask.sum()}/{N}")
                
                # Check for NaN/Inf
                print(f"  NaN in means: {torch.isnan(means_deformed).any()}")
                print(f"  NaN in scales: {torch.isnan(scales_deformed).any()}")
                print(f"  NaN in quats: {torch.isnan(quats_deformed).any()}")
                print(f"  NaN in opacities: {torch.isnan(opacities_deformed).any()}")
                
                # Check activated values
                scales_activated = torch.exp(scales_deformed)
                opacities_activated = torch.sigmoid(opacities_deformed)
                print(f"  scales_activated range: [{scales_activated.min():.6f}, {scales_activated.max():.6f}]")
                print(f"  opacities_activated range: [{opacities_activated.min():.6f}, {opacities_activated.max():.6f}]")
                self._deform_debug_logged = True
            
            # Track deformation magnitude
            with torch.no_grad():
                self._deformation_accum[deform_mask] += torch.abs(
                    means_deformed - means_canonical[deform_mask]
                )
            
            # Clone canonical parameters and update deformed subset
            means_final = means_canonical.clone()
            means_final[deform_mask] = means_deformed
            
            scales_final = scales_raw.clone()
            scales_final[deform_mask] = scales_deformed
            
            quats_final = quats_raw.clone()
            quats_final[deform_mask] = quats_deformed
            
            opacities_final = opacities_raw.clone()
            # Deformation returns [N_masked, 1], squeeze to [N_masked] before assigning
            opacities_final[deform_mask] = opacities_deformed.squeeze(-1)
            
            # Now apply activations
            means = means_final
            scales = torch.exp(scales_final)
            quats = quats_final  # Will be normalized by rasterization
            opacities = torch.sigmoid(opacities_final)
        else:
            # Coarse stage OR no deformation: use canonical parameters
            means = means_canonical
            scales = torch.exp(scales_raw)
            quats = quats_raw
            opacities = torch.sigmoid(opacities_raw)
        
        # Get colors (SH coefficients)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        # Set rendering mode
        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        
        # Get rendering parameters from kwargs or config
        sh_degree = kwargs.pop("sh_degree", self.cfg.sh_degree)
        near_plane = kwargs.pop("near_plane", self.cfg.near_plane)
        far_plane = kwargs.pop("far_plane", self.cfg.far_plane)
        render_mode = kwargs.pop("render_mode", "RGB+ED")  # Always render depth for endoscopy
        
        # Call gsplat's rasterization
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [B, 4, 4]
            Ks=Ks,  # [B, 3, 3]
            width=width,
            height=height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=sh_degree,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=camera_model,
            render_mode=render_mode,
            **kwargs,
        )
        
        # CRITICAL: DO NOT mask the rendered output here!
        # Masking should ONLY happen in loss computation.
        # If we mask here, gradients won't flow to tool-region Gaussians,
        # and they'll never learn to render tissue instead of tools.
        # See MASK_USAGE_FINE_COMB_COMPARISON.md for detailed explanation.
        
        return render_colors, render_alphas, info
    
    def compute_depth_loss(
        self,
        rendered_depth: Tensor,
        gt_depth: Tensor,
        mask: Tensor,
        mode: str,
    ) -> Tensor:
        """Compute endoscopy-specific depth loss
        
        Args:
            rendered_depth: Rendered depth [B, H, W]
            gt_depth: Ground truth depth [B, H, W]
            mask: Valid pixel mask [B, H, W]
            mode: 'binocular' or 'monocular'
            
        Returns:
            depth_loss: Scalar loss tensor
        """
        # Check if valid depth exists
        if (gt_depth != 0).sum() < 10:
            return torch.tensor(0.0, device=self.device)
        
        if mode == 'binocular':
            # Binocular: L1 loss in disparity space
            rendered_depth_inv = torch.zeros_like(rendered_depth)
            gt_depth_inv = torch.zeros_like(gt_depth)
            
            valid_mask = rendered_depth != 0
            rendered_depth_inv[valid_mask] = 1.0 / rendered_depth[valid_mask]
            
            valid_mask = gt_depth != 0
            gt_depth_inv[valid_mask] = 1.0 / gt_depth[valid_mask]
            
            return l1_loss(rendered_depth_inv, gt_depth_inv, mask)
            
        elif mode == 'monocular':
            # Monocular: Pearson correlation loss
            rendered_flat = rendered_depth.reshape(-1, 1)
            gt_flat = gt_depth.reshape(-1, 1)
            mask_flat = mask.reshape(-1)
            
            rendered_masked = rendered_flat[mask_flat != 0, :]
            gt_masked = gt_flat[mask_flat != 0, :]
            
            if rendered_masked.numel() == 0 or gt_masked.numel() == 0:
                return torch.tensor(0.0, device=self.device)
            
            corr = pearson_corrcoef(gt_masked, rendered_masked)
            return self.cfg.pearson_lambda * (1.0 - corr)
        else:
            raise ValueError(f"Unknown depth mode: {mode}")
    
    def train(self):
        """Main training loop with one-stage or two-stage support
        
        Supports:
        - One-stage training: Standard gsplat training
        - Two-stage training: Coarse (fixed cam) → Fine (random sampling)
        """
        cfg = self.cfg
        
        if cfg.two_stage_training:
            print("\n[Phase 5] Starting TWO-STAGE training:")
            print(f"  Coarse stage: {cfg.coarse_iterations} iterations (fixed camera)")
            print(f"  Fine stage:   {cfg.fine_iterations} iterations (random sampling)")
            
            # Stage 1: Coarse
            print(f"\n{'='*70}")
            print("  COARSE STAGE")
            print(f"{'='*70}")
            self._train_stage(
                stage="coarse",
                num_iterations=cfg.coarse_iterations,
            )
            
            # Stage 2: Fine
            print(f"\n{'='*70}")
            print("  FINE STAGE")
            print(f"{'='*70}")
            self._train_stage(
                stage="fine",
                num_iterations=cfg.fine_iterations,
            )
        else:
            print("\n[Phase 5] Starting ONE-STAGE training:")
            print(f"  Iterations: {cfg.max_steps}")
            
            # Single stage training
            self._train_stage(
                stage="fine",  # Use fine-stage parameters
                num_iterations=cfg.max_steps,
            )
        
        print(f"\n{'='*70}")
        print("  TRAINING COMPLETE! 🎉")
        print(f"{'='*70}\n")
    
    def _train_stage(self, stage: str, num_iterations: int):
        """Train for a single stage
        
        Args:
            stage: 'coarse' or 'fine'
            num_iterations: Number of iterations for this stage
        """
        cfg = self.cfg
        device = self.device
        
        # Setup data loader
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=(stage == "fine"),  # Shuffle for fine, not for coarse
            num_workers=0,  # Single worker to avoid multiprocessing issues
            pin_memory=True,
        )
        
        # Get stage-specific thresholds
        if stage == "coarse":
            opacity_threshold = cfg.opacity_threshold_coarse
            densify_threshold = cfg.densify_grad_threshold_coarse
        else:
            # Fine stage has decaying thresholds
            opacity_threshold_init = cfg.opacity_threshold_fine_init
            opacity_threshold_after = cfg.opacity_threshold_fine_after
            densify_threshold_init = cfg.densify_grad_threshold_fine_init
            densify_threshold_after = cfg.densify_grad_threshold_after
        
        # Training loop
        trainloader_iter = iter(trainloader)
        pbar = tqdm.tqdm(range(num_iterations), desc=f"{stage.capitalize()} stage")
        
        for step in pbar:
            # Get data
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            
            # Stage-specific camera selection
            if stage == "coarse":
                # Coarse: Always use first training frame
                data = self.trainset[0]
                # Batch it
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.unsqueeze(0)
            
            # Move data to device
            camtoworld = data["camtoworld"].to(device)  # [B, 4, 4]
            K = data["K"].to(device)  # [B, 3, 3]
            image_gt = data["image"].to(device)  # [B, H, W, 3]
            depth_gt = data["depth"].to(device)  # [B, H, W]
            mask = data["mask"].to(device)  # [B, H, W]
            time = data["time"].to(device) if "time" in data else None  # [B]
            
            # ===== PRIORITY 4: Apply mask to GT (SurgicalGaussian approach) =====
            # This is CRITICAL to prevent network from learning tool appearance!
            # Masks applied to BOTH GT and rendered images so they're in same space.
            # CONFIGURABLE: Only if use_masks=True
            if cfg.use_masks:
                mask_expanded = mask.unsqueeze(-1)  # [B, H, W, 1]
                image_gt = image_gt * mask_expanded  # Zero out tool regions in GT
                depth_gt = depth_gt * mask  # Zero out tool depth in GT
            else:
                # Full scene mode: Don't mask GT, reconstruct everything
                mask_expanded = torch.ones_like(mask).unsqueeze(-1)  # All ones (no masking)
            # ====================================================================
            
            height, width = image_gt.shape[1:3]
            
            # SH degree scheduling
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            
            # Forward pass - render
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworld,
                Ks=K,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                time_idx=time,  # Phase 6: Time for deformation
                stage=stage,  # Phase 6: Pass stage (coarse/fine)
                render_mode="RGB+ED",  # RGB + Expected Depth
            )
            
            # Split RGB and depth
            image_rendered = renders[..., :3]  # [B, H, W, 3]
            depth_rendered = renders[..., 3:4].squeeze(-1)  # [B, H, W]
            
            # ===== PRIORITY 4: Apply mask to rendered output as well =====
            # Following SurgicalGaussian: mask BOTH GT and rendered
            # This ensures loss is computed only on tissue regions
            # CONFIGURABLE: Only if use_masks=True
            if cfg.use_masks:
                image_rendered = image_rendered * mask_expanded
                depth_rendered = depth_rendered * mask
            # =============================================================
            
            # Debug: Check for NaN/Inf in rendered outputs
            if torch.isnan(image_rendered).any() or torch.isinf(image_rendered).any():
                print(f"[WARNING] NaN/Inf detected in rendered image at step {step}!")
                print(f"  NaN count: {torch.isnan(image_rendered).sum()}")
                print(f"  Inf count: {torch.isinf(image_rendered).sum()}")
            
            if stage == "fine" and step % 100 == 0 and hasattr(self, 'deform_net') and self.deform_net is not None:
                print(f"\n[MONITOR] Step {step}:")
                print(f"  Image rendered range: [{image_rendered.min():.3f}, {image_rendered.max():.3f}]")
                print("  Gaussian parameters:")
                print(f"    means range: [{self.splats['means'].min():.3f}, {self.splats['means'].max():.3f}]")
                print(f"    scales (raw) range: [{self.splats['scales'].min():.3f}, {self.splats['scales'].max():.3f}]")
                print(f"    opacities (raw) range: [{self.splats['opacities'].min():.3f}, {self.splats['opacities'].max():.3f}]")
                print(f"    opacities (activated) range: [{torch.sigmoid(self.splats['opacities']).min():.6f}, {torch.sigmoid(self.splats['opacities']).max():.6f}]")
                
                # Monitor deformation magnitude
                if hasattr(self, '_deformation_accum'):
                    deform_mag = self._deformation_accum.norm(dim=-1)
                    deform_count = (deform_mag > 0).sum()
                    print("  Deformation statistics:")
                    print(f"    Accumulated magnitude: mean={deform_mag.mean():.6f}, max={deform_mag.max():.6f}")
                    print(f"    Active deformations: {deform_count}/{len(deform_mag)}")
                    print(f"    Deformable Gaussians: {self._deformation_table.sum()}/{len(self._deformation_table)}")
                
                # Monitor gradient norms for deformation network
                total_grad_norm = 0.0
                mlp_grad_norm = 0.0
                grid_grad_norm = 0.0
                for name, param in self.deform_net.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.norm().item()
                        total_grad_norm += param_norm ** 2
                        if 'grid' in name:
                            grid_grad_norm += param_norm ** 2
                        else:
                            mlp_grad_norm += param_norm ** 2
                
                total_grad_norm = total_grad_norm ** 0.5
                mlp_grad_norm = mlp_grad_norm ** 0.5
                grid_grad_norm = grid_grad_norm ** 0.5
                print("  Deformation gradient norms:")
                print(f"    Total: {total_grad_norm:.6f}")
                print(f"    MLP: {mlp_grad_norm:.6f}")
                print(f"    Grid: {grid_grad_norm:.6f}")
            
            # ===== Loss Computation =====
            
            # 1. L1 loss (with masks)
            # Need to permute for loss functions: [B, H, W, 3] → [B, 3, H, W]
            image_rendered_p = image_rendered.permute(0, 3, 1, 2)
            image_gt_p = image_gt.permute(0, 3, 1, 2)
            mask_p = mask.unsqueeze(1)  # [B, 1, H, W]
            
            l1_loss_val = l1_loss(image_rendered_p, image_gt_p, mask_p)
            
            # 2. Depth loss (endoscopy-specific)
            depth_rendered_p = depth_rendered.unsqueeze(1)  # [B, 1, H, W]
            depth_gt_p = depth_gt.unsqueeze(1)  # [B, 1, H, W]
            
            # Separate masking for depth: can use depth in tool regions for 3D constraints
            if cfg.depth_supervise_tools:
                # Use depth everywhere (including tools) for better 3D structure
                depth_mask = None
            else:
                # Current behavior: exclude tools from depth loss too
                depth_mask = mask
            
            depth_loss_val = self.compute_depth_loss(
                rendered_depth=depth_rendered,
                gt_depth=depth_gt,
                mask=depth_mask,  # Different from RGB mask when depth_supervise_tools=True
                mode=cfg.depth_mode,
            )
            
            # 3. TV loss (targeted regularization - Priority 2)
            if cfg.tv_on_occluded_only and self.invisible_mask is not None:
                # Apply TV loss only to chronically occluded regions
                # Reshape invisible mask to match batch dimension
                inpaint_mask_batch = self.invisible_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Expand to batch size if needed
                if image_rendered_p.shape[0] > 1:
                    inpaint_mask_batch = inpaint_mask_batch.repeat(image_rendered_p.shape[0], 1, 1, 1)
                
                tv_image = compute_tv_loss_targeted(image_rendered_p, mask=inpaint_mask_batch)
                tv_depth = compute_tv_loss_targeted(depth_rendered_p, mask=inpaint_mask_batch)
                
                if step % 100 == 0 and self.world_rank == 0:
                    print(f"[Priority 2] Targeted TV loss: image={tv_image.item():.6f}, depth={tv_depth.item():.6f}")
            else:
                # Original: apply TV to entire image
                tv_image = TV_loss(image_rendered_p)
                tv_depth = TV_loss(depth_rendered_p)
            
            tv_loss_val = cfg.tv_lambda * (tv_image + tv_depth)
            
            # 4. SSIM loss (optional)
            ssim_loss_val = torch.tensor(0.0, device=device)
            if cfg.ssim_lambda > 0:
                ssim_val = ssim(image_rendered_p, image_gt_p)
                ssim_loss_val = cfg.ssim_lambda * (1.0 - ssim_val)
            
            # 5. LPIPS loss (optional)
            lpips_loss_val = torch.tensor(0.0, device=device)
            if cfg.lpips_lambda > 0:
                lpips_val = lpips_loss(image_rendered_p, image_gt_p, self.lpips)
                lpips_loss_val = cfg.lpips_lambda * lpips_val
            
            # 6. Deformation regularization (Phase 6)
            deform_reg_loss = torch.tensor(0.0, device=device)
            if hasattr(self, 'deform_net') and self.deform_net is not None:
                if (cfg.time_smoothness_weight > 0 or cfg.l1_time_planes_weight > 0 or cfg.plane_tv_weight > 0):
                    # Import regulation from gaussian_model
                    from scene.regulation import compute_plane_smoothness
                    
                    multi_res_grids = self.deform_net.deformation_net.grid.grids
                    plane_tv = 0.0
                    time_smooth = 0.0
                    l1_time = 0.0
                    
                    for grids in multi_res_grids:
                        if len(grids) == 3:
                            time_grids = []
                        else:
                            time_grids_smooth = [2, 4, 5]  # Temporal grids
                            time_grids_spatial = [0, 1, 3]  # Spatial grids
                            
                            # Time smoothness
                            for grid_id in time_grids_smooth:
                                time_smooth += compute_plane_smoothness(grids[grid_id])
                            
                            # Plane TV
                            for grid_id in time_grids_spatial:
                                plane_tv += compute_plane_smoothness(grids[grid_id])
                            
                            # L1 regularization
                            for grid_id in time_grids_smooth:
                                l1_time += torch.abs(1 - grids[grid_id]).mean()
                    
                    deform_reg_loss = (
                        cfg.plane_tv_weight * plane_tv +
                        cfg.time_smoothness_weight * time_smooth +
                        cfg.l1_time_planes_weight * l1_time
                    )
            
            # Total loss
            loss = (
                l1_loss_val +
                cfg.depth_lambda * depth_loss_val +
                tv_loss_val +
                ssim_loss_val +
                lpips_loss_val +
                deform_reg_loss
            )
            
            # Compute PSNR for logging
            psnr_val = psnr(image_rendered_p, image_gt_p, mask_p).mean()
            
            # Track best PSNR in fine stage for checkpoint saving
            if stage == "fine" and psnr_val.item() > self.best_psnr:
                self.best_psnr = psnr_val.item()
                self.best_psnr_step = step
                # Save best checkpoint
                if self.world_rank == 0:
                    self._save_checkpoint(step, stage, is_best=True)
            
            # ===== Backward Pass =====
            
            # Pre-backward hook for strategy
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )
            
            loss.backward()
            
            # ===== Optimizer Step =====
            
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Optimize deformation network
            for optimizer in self.deform_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # ===== Post-backward: Densification =====
            
            # Update densification thresholds for fine stage
            if stage == "fine":
                progress = step / num_iterations
                opacity_threshold = (
                    opacity_threshold_init -
                    progress * (opacity_threshold_init - opacity_threshold_after)
                )
                densify_threshold = (
                    densify_threshold_init -
                    progress * (densify_threshold_init - densify_threshold_after)
                )
            
            # Track Gaussian count before densification
            num_gs_before = len(self.splats["means"]) if hasattr(self, 'splats') else 0
            
            # Apply densification strategy
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                # MCMCStrategy needs learning rate
                lr = self.optimizers["means"].param_groups[0]["lr"]
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=lr,
                )
            
            # Phase 6: Update deformation table if Gaussians were added/removed
            if hasattr(self, '_deformation_table'):
                num_gs_after = len(self.splats["means"])
                if num_gs_after != num_gs_before:
                    # Resize deformation table and accumulator
                    if num_gs_after > num_gs_before:
                        # Gaussians added - mark new ones as deformable
                        num_new = num_gs_after - num_gs_before
                        new_mask = torch.ones(num_new, dtype=torch.bool, device=self.device)
                        self._deformation_table = torch.cat([self._deformation_table, new_mask])
                        
                        new_accum = torch.zeros((num_new, 3), device=self.device)
                        self._deformation_accum = torch.cat([self._deformation_accum, new_accum])
                    else:
                        # Gaussians removed - this shouldn't happen in current strategy
                        # But handle it anyway by resizing to current size
                        self._deformation_table = self._deformation_table[:num_gs_after]
                        self._deformation_accum = self._deformation_accum[:num_gs_after]
            
            # Phase 7: Tool Masking - Update deformation table based on tool masks
            if (cfg.enable_tool_masking and 
                stage == "fine" and 
                step % cfg.tool_mask_update_interval == 0 and 
                step > 0):
                self.update_deformation_table_with_tool_masks()
            
            # ===== Logging =====
            
            if step % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "psnr": f"{psnr_val.item():.2f}",
                    "l1": f"{l1_loss_val.item():.4f}",
                    "depth": f"{depth_loss_val.item():.4f}",
                    "GS": len(self.splats["means"]),
                })
            
            # Tensorboard logging
            if self.world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar(f"{stage}/loss", loss.item(), step)
                self.writer.add_scalar(f"{stage}/l1_loss", l1_loss_val.item(), step)
                self.writer.add_scalar(f"{stage}/depth_loss", depth_loss_val.item(), step)
                self.writer.add_scalar(f"{stage}/tv_loss", tv_loss_val.item(), step)
                self.writer.add_scalar(f"{stage}/psnr", psnr_val.item(), step)
                self.writer.add_scalar(f"{stage}/num_GS", len(self.splats["means"]), step)
                
                # Log deformation regularization if active
                if deform_reg_loss.item() > 0:
                    self.writer.add_scalar(f"{stage}/deform_reg_loss", deform_reg_loss.item(), step)
                
                # Log deformation magnitude statistics
                if hasattr(self, '_deformation_accum') and stage == "fine":
                    deform_mag = self._deformation_accum.norm(dim=-1)
                    self.writer.add_scalar(f"{stage}/deform_magnitude_mean", deform_mag.mean().item(), step)
                    self.writer.add_scalar(f"{stage}/deform_magnitude_max", deform_mag.max().item(), step)
                
                self.writer.flush()
            
            # Save checkpoints
            if step in [i - 1 for i in cfg.save_steps] or step == num_iterations - 1:
                self._save_checkpoint(step, stage)
            
            # Run evaluation
            if step in [i - 1 for i in cfg.eval_steps] or step == num_iterations - 1:
                self.eval(step, stage=f"{stage}_val")
        
        pbar.close()
        print(f"[Phase 5] {stage.capitalize()} stage complete!")
        
        # Report best PSNR for fine stage
        if stage == "fine" and self.world_rank == 0:
            print(f"\n[BEST PSNR] Fine stage best: {self.best_psnr:.3f} at step {self.best_psnr_step}")
            print(f"[BEST PSNR] Checkpoint saved as: {self.ckpt_dir}/fine_best_psnr.pt")
    
    def _save_checkpoint(self, step: int, stage: str, is_best: bool = False):
        """Save model checkpoint
        
        Args:
            step: Current training step
            stage: Current stage ('coarse' or 'fine')
            is_best: Whether this is the best PSNR checkpoint
        """
        if self.world_rank != 0:
            return  # Only rank 0 saves
        
        if is_best:
            checkpoint_path = f"{self.ckpt_dir}/{stage}_best_psnr.pt"
        else:
            checkpoint_path = f"{self.ckpt_dir}/{stage}_step{step:05d}.pt"
        
        data = {
            "step": step,
            "stage": stage,
            "splats": self.splats.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "strategy_state": self.strategy_state,
        }
        
        # Add best PSNR tracking info
        if is_best:
            data["best_psnr"] = self.best_psnr
            data["best_psnr_step"] = self.best_psnr_step
        
        # Phase 6: Save deformation network
        if hasattr(self, 'deform_net') and self.deform_net is not None:
            data["deform_net"] = self.deform_net.state_dict()
            data["deform_optimizers"] = [opt.state_dict() for opt in self.deform_optimizers]
            if not is_best:  # Only print once per save
                print("[Phase 6] Deformation network included in checkpoint")
        
        torch.save(data, checkpoint_path)
        
        if is_best:
            print(f"[BEST] New best PSNR: {self.best_psnr:.3f} at step {step} - Checkpoint saved: {checkpoint_path}")
        else:
            print(f"[Phase 5] Checkpoint saved: {checkpoint_path}")
    
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Evaluation on validation set
        
        Args:
            step: Current training step
            stage: Stage name for logging (e.g., "val", "coarse_val", "fine_val")
        """
        print(f"\n[Phase 5] Running evaluation at step {step}...")
        cfg = self.cfg
        device = self.device
        
        # Create validation data loader
        valloader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        
        metrics = defaultdict(list)
        
        for i, data in enumerate(valloader):
            camtoworld = data["camtoworld"].to(device)
            K = data["K"].to(device)
            image_gt = data["image"].to(device)
            depth_gt = data["depth"].to(device)
            mask = data["mask"].to(device)
            time = data["time"].to(device) if "time" in data else None
            
            height, width = image_gt.shape[1:3]
            
            # Render (with deformation if available)
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworld,
                Ks=K,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                time_idx=time,  # Pass time for deformation
                stage="fine",  # Use fine stage (with deformation) for eval
                render_mode="RGB",  # RGB only for metrics
            )
            
            image_rendered = torch.clamp(renders, 0.0, 1.0)
            
            # Save rendered images
            if self.world_rank == 0 and i < 5:  # Save first 5
                canvas = torch.cat([image_gt, image_rendered], dim=2)  # Side by side
                canvas = canvas.squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step:05d}_{i:03d}.png",
                    canvas,
                )
            
            # Compute metrics
            image_rendered_p = image_rendered.permute(0, 3, 1, 2)
            image_gt_p = image_gt.permute(0, 3, 1, 2)
            
            metrics["psnr"].append(self.psnr(image_rendered_p, image_gt_p))
            metrics["ssim"].append(self.ssim(image_rendered_p, image_gt_p))
            metrics["lpips"].append(self.lpips(image_rendered_p, image_gt_p))
        
        # Aggregate metrics
        if self.world_rank == 0:
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats["num_GS"] = len(self.splats["means"])
            
            print("[Phase 5] Evaluation Results:")
            print(f"  PSNR:  {stats['psnr']:.3f}")
            print(f"  SSIM:  {stats['ssim']:.4f}")
            print(f"  LPIPS: {stats['lpips']:.4f}")
            print(f"  Num GS: {stats['num_GS']}")
            
            # Save stats
            with open(f"{self.stats_dir}/{stage}_step{step:05d}.json", "w") as f:
                json.dump(stats, f)
            
            # Log to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"eval/{stage}/{k}", v, step)
            self.writer.flush()
        
        print("[Phase 5] Evaluation complete!")
    
    def update_deformation_table_with_tool_masks(self):
        """Update deformation table based on tool masks across all training frames
        
        Strategy:
        1. Project all Gaussians to each training frame
        2. Check which Gaussians appear in tool regions (mask == 0)
        3. Accumulate hits across frames
        4. Mark Gaussians appearing in tool regions >= threshold times as non-deformable
        
        This prevents tool-region Gaussians from deforming, enabling better
        reconstruction of tissue hidden under tools.
        """
        if not hasattr(self, 'deform_net') or self.deform_net is None:
            print("[Tool Masking] Warning: Deformation network not active, skipping update")
            return
        
        cfg = self.cfg
        device = self.device
        
        print("\n[Tool Masking] Updating deformation table...")
        
        # Get current Gaussian positions (canonical, before deformation)
        means = self.splats["means"].detach()  # [N, 3]
        N = means.shape[0]
        
        # Accumulator: count how many frames each Gaussian appears in tool region
        tool_region_hits = torch.zeros(N, device=device, dtype=torch.int32)
        
        # Get all training frames
        num_frames = len(self.trainset)
        
        # Project to each training frame
        with torch.no_grad():
            for frame_idx in range(num_frames):
                # Get frame data
                data = self.trainset[frame_idx]
                c2w = data['camtoworld'].to(device)  # [4, 4]
                K = data['K'].to(device)  # [3, 3]
                mask = data['mask'].to(device)  # [H, W]
                
                height, width = mask.shape
                
                # Project Gaussians to 2D
                means_2d = project_gaussians_to_image(means, c2w, K)  # [N, 2]
                
                # Check which are in tool regions
                in_tool = check_gaussians_in_tool_region(means_2d, mask, width, height)  # [N]
                
                # Accumulate hits
                tool_region_hits += in_tool.int()
        
        # Update deformation table
        # Mark as non-deformable if appeared in tool region >= threshold times
        non_deformable_mask = tool_region_hits >= cfg.tool_mask_threshold
        
        # Update deformation table
        # True = can deform (tissue), False = cannot deform (tool)
        self._deformation_table = ~non_deformable_mask
        
        # Report statistics
        num_deformable = self._deformation_table.sum().item()
        num_non_deformable = (~self._deformation_table).sum().item()
        pct_deformable = 100.0 * num_deformable / N
        
        print(f"[Tool Masking] Projected Gaussians to {num_frames} training frames")
        print("[Tool Masking] Deformation table updated:")
        print(f"  Deformable (tissue):     {num_deformable}/{N} ({pct_deformable:.1f}%)")
        print(f"  Non-deformable (tool):   {num_non_deformable}/{N} ({100-pct_deformable:.1f}%)")
        print(f"  Threshold: {cfg.tool_mask_threshold} frame(s)")


# ============================================================================
# Helper Functions (Phase 3)
# ============================================================================

def project_gaussians_to_image(
    means: Tensor,
    c2w: Tensor,
    K: Tensor,
) -> Tensor:
    """Project 3D Gaussian means to 2D image coordinates
    
    Args:
        means: Gaussian positions in world space [N, 3]
        c2w: Camera-to-world transform [4, 4]
        K: Camera intrinsics [3, 3]
        
    Returns:
        means_2d: 2D image coordinates [N, 2] (u, v)
    """
    # Convert to homogeneous coordinates [N, 4]
    N = means.shape[0]
    means_homo = torch.cat([means, torch.ones(N, 1, device=means.device)], dim=-1)  # [N, 4]
    
    # Transform to camera space: world-to-camera (inverse of c2w)
    w2c = torch.inverse(c2w)  # [4, 4]
    means_cam = (w2c @ means_homo.T).T  # [N, 4]
    
    # Extract 3D camera coordinates
    x = means_cam[:, 0]  # [N]
    y = means_cam[:, 1]  # [N]
    z = means_cam[:, 2]  # [N]
    
    # Project to image plane
    # u = fx * x/z + cx
    # v = fy * y/z + cy
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Avoid division by zero
    z_safe = torch.clamp(z, min=1e-6)
    
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy
    
    means_2d = torch.stack([u, v], dim=-1)  # [N, 2]
    
    return means_2d


def check_gaussians_in_tool_region(
    means_2d: Tensor,
    mask: Tensor,
    width: int,
    height: int,
) -> Tensor:
    """Check which Gaussians project into tool regions
    
    Args:
        means_2d: 2D image coordinates [N, 2] (u, v)
        mask: Tool mask [H, W] where 0=tool, 1=tissue
        width: Image width
        height: Image height
        
    Returns:
        in_tool: Boolean tensor [N] indicating if Gaussian is in tool region
    """
    N = means_2d.shape[0]
    
    # Clamp coordinates to image bounds
    u = torch.clamp(means_2d[:, 0], 0, width - 1)
    v = torch.clamp(means_2d[:, 1], 0, height - 1)
    
    # Convert to integer pixel coordinates
    u_int = u.long()
    v_int = v.long()
    
    # Check if out of bounds (will be marked as not in tool region)
    valid_mask = (
        (means_2d[:, 0] >= 0) & 
        (means_2d[:, 0] < width) & 
        (means_2d[:, 1] >= 0) & 
        (means_2d[:, 1] < height)
    )
    
    # Look up mask values: 0 = tool, 1 = tissue
    # mask shape is [H, W], access as mask[v, u]
    mask_values = mask[v_int, u_int]
    
    # Gaussian is in tool region if:
    # 1. It's within image bounds AND
    # 2. Mask value is 0 (tool)
    in_tool = valid_mask & (mask_values == 0)
    
    return in_tool


def knn(x: Tensor, K: int = 4) -> Tensor:
    """K-nearest neighbors distance computation
    
    Args:
        x: Points tensor [N, 3]
        K: Number of nearest neighbors
        
    Returns:
        distances: [N, K] tensor of distances to K nearest neighbors
    """
    from sklearn.neighbors import NearestNeighbors
    
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB to spherical harmonics (SH) coefficients
    
    Args:
        rgb: RGB values [N, 3] in range [0, 1]
        
    Returns:
        sh0: SH coefficients [N, 3]
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def create_splats_with_optimizers(
    parser: EndoNeRFParser,
    cfg: EndoConfig,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Create Gaussian splats and optimizers from point cloud
    
    Adapted from gsplat's create_splats_with_optimizers for EndoNeRF data.
    
    Args:
        parser: EndoNeRFParser with point cloud data
        cfg: EndoConfig with training parameters
        device: Device to create tensors on
        world_rank: Rank for distributed training
        world_size: Total number of processes
        
    Returns:
        splats: ParameterDict with means, scales, quats, opacities, sh0, shN
        optimizers: Dict of optimizers for each parameter
    """
    print(f"[Phase 3] Initializing {parser.points.shape[0]} Gaussians from point cloud...")
    
    # Load points and colors from parser
    if cfg.init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        print("[Phase 3] Using SfM point cloud initialization")
    elif cfg.init_type == "random":
        points = cfg.init_extent * parser.scene_scale * (
            torch.rand((cfg.init_num_pts, 3)) * 2 - 1
        )
        rgbs = torch.rand((cfg.init_num_pts, 3))
        print(f"[Phase 3] Using random initialization with {cfg.init_num_pts} points")
    else:
        raise ValueError(f"Unknown init_type: {cfg.init_type}")
    
    # Initialize scales using KNN
    print("[Phase 3] Computing KNN for scale initialization...")
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    # Clamp to avoid log(0) = -inf for duplicate points
    dist_avg = torch.clamp_min(dist_avg, 1e-7)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    
    # Distribute across ranks (for distributed training)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]
    
    N = points.shape[0]
    print(f"[Phase 3] Rank {world_rank}: {N} Gaussians")
    
    # Initialize orientations (quaternions) randomly
    quats = torch.rand((N, 4))  # [N, 4]
    
    # Initialize opacities
    opacities = torch.logit(torch.full((N,), cfg.init_opa))  # [N,]
    
    # Convert RGB to SH coefficients
    colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    
    # Create parameters list with learning rates
    scene_scale = parser.scene_scale * cfg.global_scale
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), cfg.means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), cfg.scales_lr),
        ("quats", torch.nn.Parameter(quats), cfg.quats_lr),
        ("opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), cfg.sh0_lr),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), cfg.shN_lr),
    ]
    
    # Create ParameterDict
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    
    # Scale learning rate based on batch size
    # Reference: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    BS = cfg.batch_size * world_size
    
    # Choose optimizer class
    if cfg.sparse_grad:
        optimizer_class = torch.optim.SparseAdam
        print("[Phase 3] Using SparseAdam optimizer")
    elif cfg.visible_adam:
        optimizer_class = SelectiveAdam
        print("[Phase 3] Using SelectiveAdam optimizer")
    else:
        optimizer_class = torch.optim.Adam
        print("[Phase 3] Using Adam optimizer")
    
    # Create optimizers for each parameter
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(0.9 ** BS, 0.999 ** BS),
        )
        for name, _, lr in params
    }
    
    print("[Phase 3] ✓ Splats initialized:")
    print(f"[Phase 3]   means:     {splats['means'].shape}")
    print(f"[Phase 3]   scales:    {splats['scales'].shape}")
    print(f"[Phase 3]   quats:     {splats['quats'].shape}")
    print(f"[Phase 3]   opacities: {splats['opacities'].shape}")
    print(f"[Phase 3]   sh0:       {splats['sh0'].shape}")
    print(f"[Phase 3]   shN:       {splats['shN'].shape}")
    
    return splats, optimizers


def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# Priority 2: Accurate Mask + Targeted TV Loss Helper Functions
# ============================================================================

def create_invisible_mask_from_paths(mask_paths: List[str], device: str = "cuda") -> Tensor:
    """
    Create invisible mask from list of mask file paths.
    
    This creates a union of all tool masks across all frames to identify
    regions that are chronically occluded (tool paths).
    
    Args:
        mask_paths: List of paths to mask PNG files
        device: Device to create tensor on
        
    Returns:
        invisible_mask: [H, W] tensor where 1=invisible (always occluded), 0=visible
    """
    from PIL import Image
    
    inpaint_mask = None
    
    for path in tqdm.tqdm(mask_paths, desc="Creating invisible mask"):
        mask = np.array(Image.open(path)) / 255.0
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # Take first channel
        
        # Invert: 1 where tool is, 0 where tissue is
        tool_mask = 1.0 - mask
        
        # Accumulate union
        if inpaint_mask is None:
            inpaint_mask = tool_mask.astype(np.float32)
        else:
            inpaint_mask = np.clip(inpaint_mask + tool_mask, 0, 1)
    
    return torch.from_numpy(inpaint_mask).float().to(device)


def dilate_invisible_mask(mask: Tensor, kernel_size: int = 5, iterations: int = 2) -> Tensor:
    """
    Dilate invisible mask using morphological dilation to add safety margin.
    
    Args:
        mask: [H, W] tensor in [0, 1]
        kernel_size: Size of dilation kernel
        iterations: Number of dilation iterations
        
    Returns:
        dilated_mask: [H, W] tensor in [0, 1]
    """
    import cv2
    
    # Move to CPU for OpenCV
    mask_np = mask.cpu().numpy()
    
    # Convert to uint8 for OpenCV
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Dilate
    dilated_uint8 = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    
    # Convert back to float32 tensor
    dilated = torch.from_numpy(dilated_uint8.astype(np.float32) / 255.0).to(mask.device)
    
    return dilated


def compute_tv_loss_targeted(image: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Compute total variation loss, optionally on masked region only.
    
    Args:
        image: [B, C, H, W] tensor
        mask: [B, 1, H, W] tensor (optional), where 1=compute TV, 0=ignore
        
    Returns:
        tv_loss: scalar tensor
    """
    # Compute TV differences
    tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    
    # Apply mask if provided
    if mask is not None:
        # Adjust mask dimensions for TV differences
        mask_h = mask[:, :, 1:, :]  # [B, 1, H-1, W]
        mask_w = mask[:, :, :, 1:]  # [B, 1, H, W-1]
        
        # Apply masks and sum
        tv_h_sum = (tv_h * mask_h).sum()
        tv_w_sum = (tv_w * mask_w).sum()
        
        # Count valid elements (where mask is 1)
        num_elements_h = mask_h.sum() * image.shape[1] + 1e-8
        num_elements_w = mask_w.sum() * image.shape[1] + 1e-8
        
        tv = (tv_h_sum / num_elements_h) + (tv_w_sum / num_elements_w)
    else:
        # No mask: compute on full image
        tv_h_sum = tv_h.sum()
        tv_w_sum = tv_w.sum()
        tv = (tv_h_sum + tv_w_sum) / image.numel()
    
    return tv


# ============================================================================
# Priority 1: Multi-Frame Point Cloud Initialization Helper Functions
# ============================================================================

def accumulate_multiframe_pointcloud(
    endo_dataset,
    depth_mode: str = "binocular",
    sample_rate: int = 3,
    use_masks: bool = True,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulate depth and color across ALL frames using SurgicalGaussian approach.
    
    This is the MOST CRITICAL improvement: ensures tissue under tool paths
    is represented in the initial point cloud.
    
    Args:
        endo_dataset: EndoNeRF_Dataset or SCARED_Dataset instance
        depth_mode: 'binocular' or 'monocular'
        sample_rate: Sample every Nth pixel (for memory efficiency)
        use_masks: If True, exclude tool regions; if False, include everything
        device: Device for computations
        
    Returns:
        points: [N, 3] numpy array of 3D points
        colors: [N, 3] numpy array of RGB colors (0-1 range)
    """
    from PIL import Image
    
    print("[Priority 1] Multi-frame point cloud initialization starting...")
    print(f"[Priority 1] Mode: {depth_mode}, Sample rate: {sample_rate}")
    
    # Get image dimensions
    H, W = endo_dataset.img_wh[1], endo_dataset.img_wh[0]
    
    # Initialize accumulators
    depth_all = np.zeros((H, W), dtype=np.float32)
    color_all = np.zeros((H, W, 3), dtype=np.float32)
    mask_all = np.zeros((H, W), dtype=np.float32)
    inv_mask_all = np.ones((H, W), dtype=np.float32)
    
    # Get camera intrinsics (same for all frames)
    K = endo_dataset.K
    
    print(f"[Priority 1] Accumulating across {len(endo_dataset.image_paths)} frames...")
    
    # Accumulate across all frames
    for idx in tqdm.tqdm(range(len(endo_dataset.image_paths)), desc="Accumulating frames"):
        # Load image
        image_path = endo_dataset.image_paths[idx]
        image = np.array(Image.open(image_path)) / 255.0  # [H, W, 3]
        
        # Load depth
        depth_path = endo_dataset.depth_paths[idx]
        if depth_mode == 'binocular':
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            if depth.max() > 0:
                close_depth = np.percentile(depth[depth != 0], 3.0)
                inf_depth = np.percentile(depth[depth != 0], 99.8)
                depth = np.clip(depth, close_depth, inf_depth)
        elif depth_mode == 'monocular':
            depth = np.array(Image.open(depth_path))[..., 0] / 255.0
            depth[depth != 0] = (1 / depth[depth != 0]) * 0.4
            depth[depth == 0] = depth.max() if depth.max() > 0 else 0.0
        
        # Load mask (if using masks)
        if use_masks:
            # Tissue-only mode: Exclude tool regions
            mask_path = endo_dataset.masks_paths[idx]
            mask = 1.0 - np.array(Image.open(mask_path)) / 255.0  # 1=tissue, 0=tool
            if mask.ndim == 3:
                mask = mask[..., 0]
        else:
            # Full scene mode: Include everything (no masking)
            mask = np.ones((H, W), dtype=np.float32)
        
        # Valid depth mask
        valid_depth = (depth > 0).astype(np.float32)
        
        # Combine: valid depth AND (tissue-only if masked) AND not yet seen
        mask_plus = valid_depth * mask * inv_mask_all
        
        # Accumulate (only for NEW regions)
        depth_all += depth * mask_plus
        color_all += image * mask_plus[:, :, None]
        
        # Update coverage
        mask_all = np.clip(mask_all + valid_depth * mask, 0, 1)
        inv_mask_all = 1.0 - mask_all
    
    # Compute final coverage
    coverage = (mask_all > 0).sum() / mask_all.size
    print(f"[Priority 1] Accumulation complete! Coverage: {coverage*100:.1f}%")
    
    # Convert accumulated depth/color to point cloud
    print("[Priority 1] Converting to point cloud...")
    
    # Sample pixels for efficiency
    sample_mask = np.zeros((H, W), dtype=bool)
    sample_mask[::sample_rate, ::sample_rate] = True
    
    # Valid pixels: non-zero depth AND accumulated (mask_all) AND sampled
    valid = (depth_all > 0) & (mask_all > 0) & sample_mask
    
    # Get pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_valid = u[valid]
    v_valid = v[valid]
    depth_valid = depth_all[valid]
    color_valid = color_all[valid]
    
    # Unproject to 3D using camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u_valid - cx) * depth_valid / fx
    y = (v_valid - cy) * depth_valid / fy
    z = depth_valid
    
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    colors = color_valid.astype(np.float32)
    
    print(f"[Priority 1] Generated {len(points)} points from multi-frame accumulation")
    print(f"[Priority 1]   vs. single-frame: ~{H*W//100} points (typical)")
    print(f"[Priority 1]   Improvement: {len(points) / (H*W//100):.1f}x more points")
    
    return points, colors


# ============================================================================
# Main Entry Point
# ============================================================================

def main(local_rank: int, world_rank: int, world_size: int, cfg: EndoConfig):
    """Main entry point for training
    
    Args:
        local_rank: Local GPU rank
        world_rank: Global rank
        world_size: Total number of GPUs
        cfg: Configuration
    """
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")
    
    # Save configuration
    if world_rank == 0:
        os.makedirs(cfg.result_dir, exist_ok=True)
        with open(f"{cfg.result_dir}/config.yml", "w") as f:
            yaml.dump(vars(cfg), f)
        print(f"[Phase 5] Configuration saved to {cfg.result_dir}/config.yml")
    
    # Initialize runner
    runner = EndoRunner(local_rank, world_rank, world_size, cfg)
    
    # Run training
    print("\n[Phase 5] Starting training...")
    runner.train()
    
    print("\n" + "="*70)
    print("  EndoGaussian with gsplat - Training Complete! 🎉")
    print("="*70)
    print(f"  Results saved to: {cfg.result_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Usage:
    
    # One-stage training (standard gsplat)
    python gsplat_train.py --data_dir data/EndoNeRF/pulling \
        --result_dir output/endo_one_stage \
        --max_steps 10000
    
    # Two-stage training (EndoGaussian style)  
    python gsplat_train.py --data_dir data/EndoNeRF/pulling \
        --result_dir output/endo_two_stage \
        --two_stage \
        --coarse_iterations 3000
    
    # Quick smoke test
    python gsplat_train.py --data_dir data/EndoNeRF/pulling \
        --result_dir output/smoke_test \
        --max_steps 10
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description="EndoGaussian with gsplat Training")
    parser.add_argument("--data_dir", type=str, default="data/EndoNeRF/pulling")
    parser.add_argument("--result_dir", type=str, default="output/endo_gsplat")
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--coarse_iterations", type=int, default=300)
    parser.add_argument("--depth_mode", type=str, default="binocular", choices=["binocular", "monocular"])
    parser.add_argument("--two_stage", action="store_true", help="Enable two-stage training")
    parser.add_argument("--no_deformation", action="store_true", help="Disable deformation network")
    parser.add_argument("--dataset_type", type=str, default="endonerf", choices=["endonerf", "scared"])
    
    # Masking mode arguments
    parser.add_argument("--no_masks", action="store_true", 
                       help="Disable masking - Full scene reconstruction (tissue + tools)")
    parser.add_argument("--no_multiframe_init", action="store_true",
                       help="Disable multi-frame initialization (use original single-frame)")
    parser.add_argument("--no_accurate_mask", action="store_true",
                       help="Disable accurate mask creation (Priority 2)")
    parser.add_argument("--no_mask", action="store_true", help="Disable GT masking (tools visible in training)")
    parser.add_argument("--enable_tool_masking", action="store_true", help="Enable tool masking for tissue reconstruction")
    parser.add_argument("--tool_mask_threshold", type=int, default=1, help="Min frames in tool region to mark as non-deformable")
    parser.add_argument("--tool_mask_update_interval", type=int, default=500, help="Update deformation table every N steps")
    parser.add_argument("--depth_supervise_tools", action="store_true", help="Use depth supervision in tool regions for better 3D structure")
    
    args = parser.parse_args()
    
    # Create config
    cfg = EndoConfig(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        max_steps=args.max_steps,
        coarse_iterations=args.coarse_iterations,
        fine_iterations=args.max_steps,  # Use max_steps for fine stage
        depth_mode=args.depth_mode,
        two_stage_training=args.two_stage,
        use_deformation=not args.no_deformation,
        dataset_type=args.dataset_type,
        # Masking configuration
        use_masks=not args.no_masks,  # Master switch for masking mode
        multiframe_init=not args.no_multiframe_init,  # Priority 1
        accurate_mask=not args.no_accurate_mask,  # Priority 2
        use_mask=not args.no_mask,  # Legacy compatibility
        enable_tool_masking=args.enable_tool_masking,
        tool_mask_threshold=args.tool_mask_threshold,
        tool_mask_update_interval=args.tool_mask_update_interval,
        depth_supervise_tools=args.depth_supervise_tools,
    )
    
    print("\n" + "="*70)
    print("  EndoGaussian with gsplat Backend")
    print("="*70)
    print(f"  Data: {cfg.data_dir}")
    print(f"  Output: {cfg.result_dir}")
    print(f"  Mode: {'Two-stage' if cfg.two_stage_training else 'One-stage'} training")
    print(f"  Depth: {cfg.depth_mode}")
    print(f"  Iterations: {cfg.max_steps}")
    if cfg.two_stage_training:
        print(f"    Coarse: {cfg.coarse_iterations}")
        print(f"    Fine: {cfg.fine_iterations}")
    print("\n  Reconstruction Mode:")
    if cfg.use_masks:
        print("    ✅ TISSUE-ONLY (tools removed)")
        print(f"    - Priority 1 (Multi-frame init): {cfg.multiframe_init}")
        print(f"    - Priority 2 (Accurate mask): {cfg.accurate_mask}")
        print("    - Priority 4 (GT masking): ENABLED")
    else:
        print("    🔧 FULL SCENE (tissue + tools)")
        print("    - All masking disabled")
        print("    - Tools will be reconstructed")
    print("="*70 + "\n")
    
    # Run training (single GPU for now)
    main(
        local_rank=0,
        world_rank=0,
        world_size=1,
        cfg=cfg,
    )
