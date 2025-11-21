# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Complete Unified Workflow - Single Command Execution

This script orchestrates the complete training + rendering workflow:
1. Data Accumulation: Collect all frames from dataset
2. Training: Train gsplat model on accumulated data
3. Rendering: Automatically render with trained checkpoint

All in one command!

Usage:
    python run_complete_workflow.py \
        --data_dir /path/to/EndoNeRF/pulling \
        --output_dir output/my_run \
        --training_iterations 500 \
        --num_frames 30 \
        --visualize
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
    Stage 1: Accumulate data from EndoNeRF dataset.

    Args:
        data_dir: Path to EndoNeRF dataset
        output_dir: Base output directory
        num_frames: Number of frames to accumulate (-1 for all)

    Returns:
        ingestion_dir: Path to training_ingestion directory
    """
    print(f"\n{'='*70}")
    print("  STAGE 1: DATA ACCUMULATION")
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
            pose_3x5 = poses_raw[idx]
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

    print("\n✅ Stage 1 Complete: Data Accumulation")
    print(f"  - Images: {frames_to_process}")
    print(f"  - Depth: {frames_to_process}")
    print(f"  - Masks: {frames_to_process}")
    print(f"  - poses_bounds.npy: {poses_bounds.shape}\n")

    return str(ingestion_dir)


def run_training(ingestion_dir, output_dir, training_iterations, coarse_iterations, use_masks):
    """
    Stage 2: Run gsplat training.

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
    print("  STAGE 2: TRAINING")
    print(f"{'='*70}\n")

    # Validate training script exists
    training_script = Path("training/gsplat_train.py")
    if not training_script.exists():
        print(f"\n❌ ERROR: Training script not found: {training_script}")
        print(f"  Expected at: {training_script.absolute()}")
        return None

    training_output = Path(output_dir) / "trained_model"

    cmd = [
        sys.executable,
        str(training_script),
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
    est_time = training_iterations // 100
    print(f"Estimated training time: ~{est_time} minutes\n")

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
            print("\n✅ Stage 2 Complete: Training")
            print(f"  Checkpoint: {best_ckpt}\n")
            return str(best_ckpt)
        else:
            final_ckpt = ckpt_dir / f"fine_step{training_iterations-1:05d}.pt"
            if final_ckpt.exists():
                print("\n✅ Stage 2 Complete: Training")
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


def run_inference(data_dir, checkpoint_path, num_frames=-1, visualize=True):
    """
    Stage 3: Run inference with trained checkpoint.

    Args:
        data_dir: Original data directory
        checkpoint_path: Path to trained checkpoint
        num_frames: Number of frames to render. Use -1 for continuous loop of all frames,
                    or positive integer to render that many frames once and exit
        visualize: Show visualization window

    Returns:
        success: True if successful
    """
    print(f"\n{'='*70}")
    print("  STAGE 3: INFERENCE & RENDERING")
    print(f"{'='*70}\n")

    if num_frames > 0:
        print(f"Rendering {num_frames} frames (no loop)...")
    else:
        print("Rendering ALL frames in continuous loop with your trained model...")
    print("Press ESC or close window to exit visualization\n")

    # Validate rendering script exists
    script_path = Path("dynamic_rendering_viz.py")
    if not script_path.exists():
        print(f"\n❌ ERROR: Rendering script not found: {script_path}")
        print(f"  Expected at: {script_path.absolute()}")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--data_dir",
        data_dir,
        "--checkpoint",
        checkpoint_path,
    ]

    if num_frames > 0:
        # Render specific number of frames, no loop
        cmd.extend(["--num_frames", str(num_frames)])
        cmd.append("--no-loop")
    else:
        # num_frames == -1: render all frames in continuous loop (default)
        cmd.append("--loop")

    print(f"Command: {' '.join(cmd)}\n")
    print("Starting visualization...\n")

    try:
        if visualize:
            # Run with display
            ret = subprocess.call(cmd)
        else:
            # Run without display (headless)
            ret = subprocess.call(cmd, env={**os.environ, "DISPLAY": ""})

        if ret == 0:
            print("\n✅ Stage 3 Complete: Rendering")
            return True
        else:
            print(f"\n⚠ Rendering exited with code {ret} (may be normal if window closed)")
            return True  # Still consider success

    except Exception as e:
        print(f"\n❌ Rendering failed: {e}")
        return False


def main():
    """Main entry point - orchestrates complete workflow."""
    parser = ArgumentParser(description="Complete workflow: Accumulate → Train → Render")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to EndoNeRF dataset")
    parser.add_argument(
        "--output_dir", type=str, default="output/complete_workflow", help="Output directory"
    )
    parser.add_argument("--training_iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--coarse_iterations", type=int, default=50, help="Coarse iterations")
    parser.add_argument("--num_frames", type=int, default=-1, help="Number of frames (-1 = all)")
    parser.add_argument("--no_masks", action="store_true", help="Full scene (no tool removal)")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization")
    parser.add_argument(
        "--train_only", action="store_true", help="Only run training, skip rendering"
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  COMPLETE UNIFIED WORKFLOW")
    print(f"{'='*70}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Iterations: {args.training_iterations}")
    print(f"Mode: {'Full scene' if args.no_masks else 'Tissue-only'}")
    print(f"Frames: {args.num_frames if args.num_frames > 0 else 'all'}")
    print(f"{'='*70}\n")

    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Stage 1: Data Accumulation
    ingestion_dir = accumulate_data(args.data_dir, args.output_dir, args.num_frames)

    if not ingestion_dir:
        return 1

    # Stage 2: Training
    checkpoint = run_training(
        ingestion_dir,
        args.output_dir,
        args.training_iterations,
        args.coarse_iterations,
        not args.no_masks,
    )

    if not checkpoint:
        return 1

    # Stage 3: Rendering (optional)
    if not args.train_only:
        success = run_inference(
            args.data_dir, checkpoint, args.num_frames, visualize=not args.no_visualize
        )

        if not success:
            return 1

    # Final summary
    print(f"\n{'='*70}")
    print("  WORKFLOW COMPLETE! 🎉")
    print(f"{'='*70}")
    print("\nResults:")
    print(f"  Data: {ingestion_dir}")
    print(f"  Model: {Path(args.output_dir) / 'trained_model'}")
    print(f"  Checkpoint: {checkpoint}")
    if not args.train_only:
        print("  Renders: output/rendered_dynamic/")
    print("\nTo re-run inference:")
    print("  python dynamic_rendering_viz.py \\")
    print(f"      --data_dir {args.data_dir} \\")
    print(f"      --checkpoint {checkpoint}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    exit(main())
