#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
G-SHARP End-to-End Orchestrator

Runs all five pipeline phases sequentially:

    Phase 1  Streaming DA2 + MedSAM3 inference        (Holoscan)
    Phase 2  VGGT batch camera pose estimation         (standalone)
    Phase 3  EndoNeRF format conversion                (standalone)
    Phase 4  GSplat training                           (standalone)
    Phase 5  Live render viewer                        (Holoscan)

Usage (inside the Docker container):

    python /workspace/app/run_gsharp.py \\
        --data-dir /workspace/data/my_frames \\
        --output-dir /workspace/output

    Model code is bundled under ``models/`` and ``training/``, and
    checkpoints are expected in ``assets/``. See ``assets/README.md`` for details.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
TRAINING_DIR = APP_DIR / "training"

# Default paths resolve to the bundled models/ , training/ , and assets/
# directories so the pipeline works out-of-the-box when run from the
# application directory (or its Docker image). Override via CLI for other layouts.
DEFAULTS = {
    "da2_root": str(MODELS_DIR / "depth_anything_v2"),
    "da2_checkpoint": str(APP_DIR / "assets" / "da2" / "depth_anything_v2_vits.pth"),
    "sam3_checkpoint": str(APP_DIR / "assets" / "medsam3" / "checkpoint_8_new_best.pt"),
    "train_script": str(TRAINING_DIR / "train_standalone.py"),
    "training_iterations": 1400,
    "coarse_iterations": 200,
    "da2_encoder": "vits",
    "batch_size": 30,
    "depth_scale": 100.0,
    "fps": 30,
}


def _banner(title: str, char: str = "=", width: int = 70) -> str:
    line = char * width
    return f"\n{line}\n  {title}\n{line}\n"


def run_phase(
    name: str,
    cmd: list[str],
    env_extra: dict | None = None,
    cwd: str | None = None,
) -> int:
    """Run a pipeline phase as a subprocess with timing."""
    print(_banner(name))
    print(f"  Command: {' '.join(cmd)}")
    if cwd:
        print(f"  CWD:     {cwd}")
    print()

    env = os.environ.copy()
    # Always include the app directory on PYTHONPATH so operators/ and
    # stages/ packages are importable from any subprocess.
    wf_dir = str(APP_DIR)
    pp = env.get("PYTHONPATH", "")
    if wf_dir not in pp.split(os.pathsep):
        env["PYTHONPATH"] = f"{wf_dir}:{pp}" if pp else wf_dir
    if env_extra:
        env.update(env_extra)

    start = time.time()
    ret = subprocess.call(cmd, env=env, cwd=cwd)
    elapsed = time.time() - start

    status = "PASSED" if ret == 0 else f"FAILED (exit {ret})"
    print(f"\n[{name}] {status} in {elapsed:.1f}s")
    return ret


def main():
    parser = ArgumentParser(
        description="G-SHARP End-to-End Pipeline Orchestrator",
        formatter_class=lambda prog: __import__("argparse").HelpFormatter(
            prog, max_help_position=40
        ),
    )

    # Required
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing input PNG frames")
    parser.add_argument("--output-dir", required=True,
                        help="Base output directory for all pipeline artifacts")

    # Model paths (with container defaults)
    parser.add_argument("--da2-root", default=DEFAULTS["da2_root"],
                        help="DA2 model code directory")
    parser.add_argument("--da2-checkpoint", default=DEFAULTS["da2_checkpoint"],
                        help="DA2 .pth checkpoint")
    parser.add_argument("--da2-encoder", default=DEFAULTS["da2_encoder"],
                        choices=["vits", "vitb", "vitl"])
    parser.add_argument("--sam3-checkpoint", default=DEFAULTS["sam3_checkpoint"],
                        help="MedSAM3 .pt checkpoint (empty = HuggingFace)")
    parser.add_argument("--train-script", default=DEFAULTS["train_script"],
                        help="Path to train_standalone.py")

    # Training params
    parser.add_argument("--training-iterations", type=int,
                        default=DEFAULTS["training_iterations"],
                        help="Total GSplat training iterations")
    parser.add_argument("--coarse-iterations", type=int,
                        default=DEFAULTS["coarse_iterations"],
                        help="Coarse stage iterations")
    parser.add_argument("--no-deformation", action="store_true",
                        help="Disable deformation network (static scene)")

    # Phase control
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1 (reuse existing images/depth/masks)")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip Phase 2 (reuse existing VGGT poses)")
    parser.add_argument("--skip-phase3", action="store_true",
                        help="Skip Phase 3 (reuse existing EndoNeRF dataset)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip Phase 4 (no training)")
    parser.add_argument("--skip-viewer", action="store_true",
                        help="Skip Phase 5 (no live viewer)")

    # Display & progress
    parser.add_argument("--headless", action="store_true",
                        help="Run Holoscan apps without visualization")
    parser.add_argument("--progress-file", default=None,
                        help="JSON file for progress monitor (auto-generated if not set)")

    # VGGT
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"],
                        help="VGGT batch size (frames per batch)")
    parser.add_argument("--depth-scale", type=float, default=DEFAULTS["depth_scale"],
                        help="Depth scale factor (100 = centimeters)")

    # Render viewer
    parser.add_argument("--fps", type=int, default=DEFAULTS["fps"],
                        help="Render viewer playback FPS")

    args = parser.parse_args()

    # ── Derived paths ────────────────────────────────────────────────
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    phase1_output = output_dir / "phase1_raw"
    vggt_output = output_dir / "phase2_vggt"
    endonerf_dir = output_dir / "phase3_endonerf"
    train_output = output_dir / "phase4_training"
    progress_file = args.progress_file or str(output_dir / "progress.json")

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.time()

    print(_banner("G-SHARP Scene Reconstruction Pipeline", char="█"))
    print(f"  Input frames:    {data_dir}")
    print(f"  Output:          {output_dir}")
    print(f"  Training:        {args.training_iterations} iters "
          f"({args.coarse_iterations}c + {args.training_iterations - args.coarse_iterations}f)")
    print(f"  Headless:        {args.headless}")
    print(f"  Progress file:   {progress_file}")
    print()

    # ── Phase 1: Streaming DA2 + MedSAM3 ────────────────────────────
    if not args.skip_phase1:
        phase1_cmd = [
            sys.executable, str(APP_DIR / "gsplat_scene_recon.py"),
            "--images", str(data_dir),
            "--output", str(phase1_output),
            "--da2-root", args.da2_root,
            "--da2-checkpoint", args.da2_checkpoint,
            "--da2-encoder", args.da2_encoder,
        ]
        if args.sam3_checkpoint:
            phase1_cmd.extend(["--sam3-checkpoint", args.sam3_checkpoint])
        if args.headless:
            phase1_cmd.append("--headless")

        ret = run_phase("Phase 1: DA2 + MedSAM3 Streaming Inference", phase1_cmd)
        if ret != 0:
            print("\n[Pipeline] ABORTED: Phase 1 failed.")
            return ret
    else:
        print("\n[Pipeline] Skipping Phase 1 (--skip-phase1)")

    # ── Launch progress monitor (background) ─────────────────────────
    # Runs alongside Phases 2-4 to show live progress bars via HoloViz.
    progress_proc = None
    any_batch_phase = not (args.skip_phase2 and args.skip_phase3 and args.skip_training)
    if any_batch_phase and not args.headless:
        monitor_cmd = [
            sys.executable, str(APP_DIR / "progress_monitor.py"),
            "--progress-file", progress_file,
        ]
        monitor_env = os.environ.copy()
        wf_dir = str(APP_DIR)
        pp = monitor_env.get("PYTHONPATH", "")
        if wf_dir not in pp.split(os.pathsep):
            monitor_env["PYTHONPATH"] = f"{wf_dir}:{pp}" if pp else wf_dir
        print("[Pipeline] Starting progress monitor (background)...")
        progress_proc = subprocess.Popen(monitor_cmd, env=monitor_env)

    # ── Phase 2: VGGT Batch Pose Estimation ──────────────────────────
    if not args.skip_phase2:
        phase2_cmd = [
            sys.executable, str(APP_DIR / "models" / "vggt" / "vggt_inference.py"),
            "--image-dir", str(phase1_output / "images"),
            "--output-dir", str(vggt_output),
            "--batch-size", str(args.batch_size),
            "--progress-file", progress_file,
        ]

        ret = run_phase("Phase 2: VGGT Batch Pose Estimation", phase2_cmd)
        if ret != 0:
            print("\n[Pipeline] ABORTED: Phase 2 failed.")
            return ret
    else:
        print("\n[Pipeline] Skipping Phase 2 (--skip-phase2)")

    # ── Phase 3: EndoNeRF Format Conversion ──────────────────────────
    if not args.skip_phase3:
        phase3_cmd = [
            sys.executable, str(APP_DIR / "stages" / "format_conversion.py"),
            "--phase1-dir", str(phase1_output),
            "--vggt-dir", str(vggt_output),
            "--output-dir", str(endonerf_dir),
            "--depth-scale", str(args.depth_scale),
            "--progress-file", progress_file,
        ]

        ret = run_phase("Phase 3: EndoNeRF Format Conversion", phase3_cmd)
        if ret != 0:
            print("\n[Pipeline] ABORTED: Phase 3 failed.")
            return ret
    else:
        print("\n[Pipeline] Skipping Phase 3 (--skip-phase3)")

    # ── Phase 4: GSplat Training ─────────────────────────────────────
    best_ckpt = None
    if not args.skip_training:
        train_script = Path(args.train_script).resolve()
        if not train_script.exists():
            print(f"\n[Pipeline] ERROR: Training script not found: {train_script}")
            return 1

        # train_standalone.py calls gsplat_train.py as a sibling, and
        # gsplat_train.py imports scene.* and utils.* relative to CWD.
        train_cwd = str(train_script.parent)

        phase4_cmd = [
            sys.executable, str(APP_DIR / "stages" / "train_with_progress.py"),
            "--progress-file", progress_file,
            "--train-script", str(train_script),
            "--",
            "--data_dir", str(endonerf_dir),
            "--output_dir", str(train_output),
            "--training_iterations", str(args.training_iterations),
            "--coarse_iterations", str(args.coarse_iterations),
        ]
        if args.no_deformation:
            phase4_cmd.append("--no_deformation")

        ret = run_phase("Phase 4: GSplat Training", phase4_cmd, cwd=train_cwd)
        if ret != 0:
            print("\n[Pipeline] WARNING: Phase 4 (training) failed.")
            return ret

        # Locate best checkpoint
        ckpt_dir = train_output / "trained_model" / "ckpts"
        candidate = ckpt_dir / "fine_best_psnr.pt"
        if candidate.exists():
            best_ckpt = candidate
        else:
            # Fall back to final step checkpoint
            ckpts = sorted(ckpt_dir.glob("fine_step*.pt")) if ckpt_dir.exists() else []
            if ckpts:
                best_ckpt = ckpts[-1]
    else:
        print("\n[Pipeline] Skipping Phase 4 (--skip-training)")
        # Look for existing checkpoint
        candidate = train_output / "trained_model" / "ckpts" / "fine_best_psnr.pt"
        if candidate.exists():
            best_ckpt = candidate

    # ── Stop progress monitor ────────────────────────────────────────
    if progress_proc is not None and progress_proc.poll() is None:
        print("[Pipeline] Stopping progress monitor...")
        progress_proc.send_signal(signal.SIGINT)
        try:
            progress_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            progress_proc.kill()
            progress_proc.wait()

    # ── Phase 5: Live Render Viewer ──────────────────────────────────
    if not args.skip_viewer and best_ckpt is not None:
        phase5_cmd = [
            sys.executable, str(APP_DIR / "render_viewer.py"),
            "--data-dir", str(endonerf_dir),
            "--checkpoint", str(best_ckpt),
            "--fps", str(args.fps),
        ]
        if args.headless:
            phase5_cmd.append("--headless")

        # render_viewer.py imports scene.deformation from the training/
        # directory, so we put it on PYTHONPATH.
        train_dir = str(Path(args.train_script).resolve().parent)
        pp_parts = [train_dir, str(APP_DIR)]
        existing = os.environ.get("PYTHONPATH", "")
        if existing:
            pp_parts.append(existing)
        phase5_env = {"PYTHONPATH": os.pathsep.join(pp_parts)}

        ret = run_phase("Phase 5: Live Render Viewer", phase5_cmd,
                        env_extra=phase5_env)
        if ret != 0:
            print("\n[Pipeline] WARNING: Phase 5 (viewer) exited with code", ret)
    elif args.skip_viewer:
        print("\n[Pipeline] Skipping Phase 5 (--skip-viewer)")
    elif best_ckpt is None:
        print("\n[Pipeline] Skipping Phase 5: no checkpoint found")

    # ── Summary ──────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print(_banner("G-SHARP Pipeline Complete", char="█"))
    print(f"  Total time:      {total:.1f}s ({total / 60:.1f} min)")
    print(f"  Output:          {output_dir}")
    if best_ckpt:
        print(f"  Best checkpoint: {best_ckpt}")
    print(f"  EndoNeRF data:   {endonerf_dir}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
