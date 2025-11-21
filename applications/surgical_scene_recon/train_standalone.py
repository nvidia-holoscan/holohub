# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Training Script (No Holoscan)

This script collects data and runs training without using Holoscan pipeline.
Simpler and more reliable for batch processing.

Usage:
    python train_standalone.py \
        --data_dir /path/to/data \
        --output_dir output/my_model \
        --training_iterations 500 \
        --num_frames 30
"""

import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def accumulate_data(data_dir, output_dir, num_frames=-1):
    """
    Accumulate data from EndoNeRF dataset and save in training_ingestion format.

    Args:
        data_dir: Path to EndoNeRF dataset
        output_dir: Base output directory
        num_frames: Number of frames to accumulate (-1 for all)

    Returns:
        ingestion_dir: Path to training_ingestion directory
    """
    print(f"\n{'='*70}")
    print("  DATA ACCUMULATION")
    print(f"{'='*70}\n")

    # Create output directories
    ingestion_dir = Path(output_dir) / "training_ingestion"
    (ingestion_dir / "images").mkdir(parents=True, exist_ok=True)
    (ingestion_dir / "depth").mkdir(parents=True, exist_ok=True)
    (ingestion_dir / "masks").mkdir(parents=True, exist_ok=True)

    print(f"Output: {ingestion_dir}")

    # Load poses
    poses_file = Path(data_dir) / "poses_bounds.npy"
    poses_arr = np.load(poses_file)
    poses_raw = poses_arr[:, :-2].reshape([-1, 3, 5])
    total_frames = poses_raw.shape[0]

    # Determine how many frames to process
    if num_frames > 0:
        frames_to_process = min(num_frames, total_frames)
    else:
        frames_to_process = total_frames

    print(f"Processing {frames_to_process} of {total_frames} frames\n")

    # Get file lists
    import glob

    image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(data_dir, "depth", "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "masks", "*.png")))

    # Validate file counts match
    if not (len(image_paths) == len(depth_paths) == len(mask_paths)):
        print("\n❌ ERROR: Mismatched file counts!")
        print(f"  Images: {len(image_paths)}")
        print(f"  Depths: {len(depth_paths)}")
        print(f"  Masks: {len(mask_paths)}")
        raise ValueError("Image, depth, and mask directories must contain the same number of files")

    # Validate we have enough data
    available_frames = len(image_paths)
    if available_frames < frames_to_process:
        print(
            f"\n⚠ WARNING: Only {available_frames} frames available, but {frames_to_process} requested"
        )
        frames_to_process = available_frames
        print(f"  Adjusting to process {frames_to_process} frames")

    if available_frames != total_frames:
        print(
            f"\n⚠ WARNING: File count ({available_frames}) doesn't match pose count ({total_frames})"
        )
        print(f"  Using minimum: {min(available_frames, total_frames)} frames")
        frames_to_process = min(frames_to_process, available_frames, total_frames)

    print(f"Validated {available_frames} frames in each directory\n")

    # Copy frames
    poses_list = []

    for idx in tqdm(range(frames_to_process), desc="Accumulating frames"):
        try:
            # Copy image
            img = Image.open(image_paths[idx])
            img.save(ingestion_dir / "images" / f"{idx:05d}.png")

            # Copy depth
            depth = Image.open(depth_paths[idx])
            depth.save(ingestion_dir / "depth" / f"{idx:05d}.png")

            # Copy mask
            mask = Image.open(mask_paths[idx])
            mask.save(ingestion_dir / "masks" / f"{idx:05d}.png")

            # Store pose (already in 3x5 format from poses_raw)
            pose_3x5 = poses_raw[idx]  # Already [3, 5] format
            poses_list.append(pose_3x5)

        except FileNotFoundError as e:
            print(f"\n❌ ERROR: File not found for frame {idx}")
            print(f"  {e}")
            raise
        except PermissionError as e:
            print(f"\n❌ ERROR: Permission denied for frame {idx}")
            print(f"  {e}")
            raise
        except Exception as e:
            print(f"\n❌ ERROR: Failed to process frame {idx}")
            print(f"  Image: {image_paths[idx]}")
            print(f"  Depth: {depth_paths[idx]}")
            print(f"  Mask: {mask_paths[idx]}")
            print(f"  Error: {e}")
            raise

    # Save poses_bounds.npy
    poses_arr = np.stack(poses_list, axis=0)
    poses_flat = poses_arr.reshape(-1, 15)
    bounds = np.array([[0.01, 1000.0]] * len(poses_list))
    poses_bounds = np.concatenate([poses_flat, bounds], axis=1)
    np.save(ingestion_dir / "poses_bounds.npy", poses_bounds)

    print("\n✅ Accumulation complete!")
    print(f"  - Images: {frames_to_process}")
    print(f"  - Depth: {frames_to_process}")
    print(f"  - Masks: {frames_to_process}")
    print(f"  - poses_bounds.npy: {poses_bounds.shape}\n")

    return str(ingestion_dir)


def run_training(ingestion_dir, output_dir, training_iterations, coarse_iterations, use_masks):
    """
    Run gsplat training.

    Args:
        ingestion_dir: Path to training_ingestion directory
        output_dir: Base output directory
        training_iterations: Total training iterations
        coarse_iterations: Coarse stage iterations
        use_masks: Tissue-only mode

    Returns:
        checkpoint_path: Path to trained checkpoint, or None if failed
    """
    print(f"\n{'='*70}")
    print("  TRAINING")
    print(f"{'='*70}\n")

    training_output = Path(output_dir) / "trained_model"

    cmd = [
        sys.executable,
        "training/gsplat_train.py",
        "--data_dir",
        str(ingestion_dir),
        "--result_dir",
        str(training_output),
        "--max_steps",
        str(training_iterations),
        "--coarse_iterations",
        str(coarse_iterations),
        "--two_stage",
        "--depth_mode",
        "binocular",
    ]

    if not use_masks:
        cmd.append("--no_masks")

    print(f"Command: {' '.join(cmd)}\n")
    print(f"Training (~{training_iterations//100} minutes)...\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(f"[Training] {line.rstrip()}")

        return_code = process.wait()

        if return_code != 0:
            print(f"\n❌ Training failed with exit code {return_code}")
            return None

        # Find checkpoint
        ckpt_dir = training_output / "ckpts"
        best_ckpt = ckpt_dir / "fine_best_psnr.pt"

        if best_ckpt.exists():
            print("\n✅ Training complete!")
            print(f"  Checkpoint: {best_ckpt}\n")
            return str(best_ckpt)
        else:
            final_ckpt = ckpt_dir / f"fine_step{training_iterations-1:05d}.pt"
            if final_ckpt.exists():
                print("\n✅ Training complete!")
                print(f"  Checkpoint: {final_ckpt}\n")
                return str(final_ckpt)
            else:
                print("\n❌ No checkpoint found")
                return None

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = ArgumentParser(description="Standalone training script (no Holoscan)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/standalone_train")
    parser.add_argument("--training_iterations", type=int, default=500)
    parser.add_argument("--coarse_iterations", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--no_masks", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  Standalone Training Script")
    print(f"{'='*70}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Iterations: {args.training_iterations}")
    print(f"Mode: {'Full scene' if args.no_masks else 'Tissue-only'}")
    print(f"{'='*70}\n")

    # Step 1: Accumulate data
    ingestion_dir = accumulate_data(args.data_dir, args.output_dir, args.num_frames)

    # Step 2: Run training
    checkpoint = run_training(
        ingestion_dir,
        args.output_dir,
        args.training_iterations,
        args.coarse_iterations,
        not args.no_masks,
    )

    if checkpoint:
        print(f"\n{'='*70}")
        print("  SUCCESS!")
        print(f"{'='*70}")
        print(f"Trained checkpoint: {checkpoint}")
        print("\nRun inference and render with:")
        print("    ./holohub run surgical_scene_recon render")
        print(f"{'='*70}\n")
        return 0
    else:
        print(f"\n{'='*70}")
        print("  FAILED!")
        print(f"{'='*70}\n")
        return 1


if __name__ == "__main__":
    exit(main())
