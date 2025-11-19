#!/usr/bin/env python3
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
Surgical Scene Reconstruction - Unified Entry Point

This script provides a unified interface for both modes:
1. Inference-only mode: Use existing offline checkpoint
2. Train-then-render mode: Collect data, train, then render

Usage:
    # Inference-only mode (existing functionality)
    python run_surgical_recon.py \
        --mode inference \
        --data_dir /path/to/EndoNeRF/pulling \
        --checkpoint /path/to/checkpoint.pt
    
    # Train-then-render mode (new functionality)
    python run_surgical_recon.py \
        --mode train \
        --data_dir /path/to/EndoNeRF/pulling \
        --output_dir output/my_run \
        --training_iterations 2000
"""

import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


def run_inference_mode(args):
    """Run inference-only mode with existing checkpoint."""
    print(f"\n{'='*70}")
    print("  MODE: INFERENCE-ONLY")
    print(f"{'='*70}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*70}\n")
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return 1
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint does not exist: {args.checkpoint}")
        return 1
    
    # Validate rendering script exists
    rendering_script = Path("test_dynamic_rendering_viz.py")
    if not rendering_script.exists():
        print(f"\n❌ ERROR: Rendering script not found: {rendering_script}")
        print(f"  Expected at: {rendering_script.absolute()}")
        return 1
    
    # Run existing dynamic rendering script
    cmd = [
        sys.executable,
        str(rendering_script),
        "--data_dir", args.data_dir,
        "--checkpoint", args.checkpoint,
        "--num_frames", str(args.num_frames),
    ]
    
    if not args.loop:
        cmd.append("--no-loop")
    
    print(f"Running: {' '.join(cmd)}\n")
    
    return subprocess.call(cmd)


def run_train_mode(args):
    """Run train-then-render mode."""
    print(f"\n{'='*70}")
    print("  MODE: TRAIN-THEN-RENDER")
    print(f"{'='*70}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training iterations: {args.training_iterations}")
    print(f"{'='*70}\n")
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return 1
    
    # Stage 1: Data collection and training
    print(f"\n{'='*70}")
    print("  STAGE 1: DATA COLLECTION & TRAINING")
    print(f"{'='*70}\n")
    
    # Validate training script exists
    training_script = Path("train_standalone.py")
    if not training_script.exists():
        print(f"\n❌ ERROR: Training script not found: {training_script}")
        print(f"  Expected at: {training_script.absolute()}")
        return 1
    
    cmd_train = [
        sys.executable,
        str(training_script),
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--training_iterations", str(args.training_iterations),
        "--coarse_iterations", str(args.coarse_iterations),
        "--num_frames", str(args.num_frames),
    ]
    
    if args.no_masks:
        cmd_train.append("--no_masks")
    
    print(f"Running: {' '.join(cmd_train)}\n")
    
    ret = subprocess.call(cmd_train)
    if ret != 0:
        print(f"\nERROR: Training failed with exit code {ret}")
        return ret
    
    # Find trained checkpoint
    checkpoint_path = Path(args.output_dir) / "trained_model" / "ckpts" / "fine_best_psnr.pt"
    
    if not checkpoint_path.exists():
        print(f"\nERROR: Trained checkpoint not found at {checkpoint_path}")
        return 1
    
    # Stage 2: Inference with trained checkpoint
    print(f"\n{'='*70}")
    print("  STAGE 2: INFERENCE WITH TRAINED CHECKPOINT")
    print(f"{'='*70}\n")
    
    # Validate rendering script exists
    rendering_script = Path("test_dynamic_rendering_viz.py")
    if not rendering_script.exists():
        print(f"\n❌ ERROR: Rendering script not found: {rendering_script}")
        print(f"  Expected at: {rendering_script.absolute()}")
        return 1
    
    cmd_render = [
        sys.executable,
        str(rendering_script),
        "--data_dir", args.data_dir,
        "--checkpoint", str(checkpoint_path),
        "--num_frames", str(args.num_frames),
    ]
    
    if not args.loop:
        cmd_render.append("--no-loop")
    
    print(f"Running: {' '.join(cmd_render)}\n")
    
    ret = subprocess.call(cmd_render)
    
    if ret == 0:
        print(f"\n{'='*70}")
        print("  SUCCESS! Train-then-render complete!")
        print(f"{'='*70}")
        print(f"Trained model: {args.output_dir}/trained_model/")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Renders: {args.output_dir}/rendered_dynamic/")
        print(f"{'='*70}\n")
    
    return ret


def main():
    """Main entry point."""
    parser = ArgumentParser(
        description="Surgical Scene Reconstruction - Unified interface for inference and training modes"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "train"],
        required=True,
        help="Mode: 'inference' (use existing checkpoint) or 'train' (train then render)"
    )
    
    # Common arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to EndoNeRF dataset directory"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=-1,
        help="Number of frames to process (default: -1 = all)"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help="Loop rendering continuously"
    )
    
    # Inference-only mode arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (required for inference mode)"
    )
    
    # Train mode arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/train_then_render",
        help="Output directory for training (train mode only)"
    )
    parser.add_argument(
        "--training_iterations",
        type=int,
        default=2000,
        help="Total training iterations (train mode only, default: 2000)"
    )
    parser.add_argument(
        "--coarse_iterations",
        type=int,
        default=200,
        help="Coarse stage iterations (train mode only, default: 200)"
    )
    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="Disable masking - full scene reconstruction (train mode only)"
    )
    
    args = parser.parse_args()
    
    # Validate mode-specific requirements
    if args.mode == "inference":
        if not args.checkpoint:
            print("ERROR: --checkpoint is required for inference mode")
            return 1
        return run_inference_mode(args)
    
    elif args.mode == "train":
        return run_train_mode(args)
    
    else:
        print(f"ERROR: Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    exit(main())
