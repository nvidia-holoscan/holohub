# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GSplat Training Wrapper with Progress Reporting

Launches ``train_standalone.py`` as a subprocess and parses its tqdm output
to write progress updates for the HoloViz progress monitor.

Usage (inside Docker):
    python -m stages.train_with_progress \
        --progress-file /workspace/datasets/progress.json \
        --train-script /workspace/gsplat_standalone/train_standalone.py \
        -- \
        --data_dir /workspace/datasets/endonerf_cmr_phase3 \
        --output_dir /workspace/datasets/phase4_gsplat_output \
        --training_iterations 1400 \
        --coarse_iterations 200
"""

from __future__ import annotations

import re
import subprocess
import sys
from argparse import ArgumentParser

from stages.progress import update_progress

# Matches tqdm lines like:
#   [Training] Coarse stage:  50%|...| 100/200 [00:06<00:06, ...]
#   [Training] Fine stage:  40%|...| 560/1400 [00:33<00:47, ...]
TQDM_RE = re.compile(r"\[Training\]\s+(Coarse|Fine)\s+stage:\s+\d+%\|.*?\|\s+(\d+)/(\d+)")

# Matches best PSNR lines like:
#   [BEST] New best PSNR: 32.979 at step 1156
BEST_PSNR_RE = re.compile(r"\[BEST\].*PSNR:\s+([\d.]+)")

# Matches the "Accumulating frames" tqdm line from data ingestion
ACCUM_RE = re.compile(r"Accumulating frames:\s+\d+%\|.*?\|\s+(\d+)/(\d+)")


def _parse_int_arg(train_args: list[str], name: str) -> int:
    for i, arg in enumerate(train_args):
        if arg == name and i + 1 < len(train_args):
            return int(train_args[i + 1])
    return 0


def run_training(
    progress_file: str,
    train_script: str,
    train_args: list[str],
) -> int:
    """Launch training and report progress.

    Returns the training process exit code.
    """
    cmd = [sys.executable, train_script] + train_args

    total_iter = _parse_int_arg(train_args, "--training_iterations")
    coarse_iter = _parse_int_arg(train_args, "--coarse_iterations")
    fine_iter = max(total_iter - coarse_iter, 0)
    grand_total = coarse_iter + fine_iter

    if grand_total == 0:
        grand_total = 1  # fallback: avoid division-by-zero

    update_progress(
        progress_file, "training", "GSplat Training", 0, grand_total, "Data ingestion...", "running"
    )

    best_psnr = 0.0
    current_global = 0

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

        # Best PSNR checkpoint (check first — can appear on same line as tqdm)
        pm = BEST_PSNR_RE.search(line)
        if pm:
            best_psnr = float(pm.group(1))

        # tqdm training progress
        m = TQDM_RE.search(line)
        if m:
            stage_name = m.group(1)
            step = int(m.group(2))
            stage_total = int(m.group(3))

            if stage_name == "Coarse":
                current_global = step
            else:
                current_global = coarse_iter + step

            psnr_str = f"  |  Best PSNR: {best_psnr:.2f}" if best_psnr > 0 else ""
            detail = f"{stage_name} {step}/{stage_total}{psnr_str}"

            update_progress(
                progress_file,
                "training",
                "GSplat Training",
                current_global,
                grand_total,
                detail,
                "running",
            )
            continue

        # Data accumulation phase (before actual training begins)
        am = ACCUM_RE.search(line)
        if am:
            update_progress(
                progress_file,
                "training",
                "GSplat Training",
                0,
                grand_total,
                f"Accumulating {am.group(1)}/{am.group(2)} frames...",
                "running",
            )

    proc.wait()

    if proc.returncode == 0:
        detail = f"Complete  |  Best PSNR: {best_psnr:.2f}" if best_psnr > 0 else "Complete"
        update_progress(
            progress_file,
            "training",
            "GSplat Training",
            grand_total,
            grand_total,
            detail,
            "complete",
        )
    else:
        update_progress(
            progress_file,
            "training",
            "GSplat Training",
            current_global,
            grand_total,
            f"Failed (exit code {proc.returncode})",
            "error",
        )

    return proc.returncode


def main():
    parser = ArgumentParser(
        description="GSplat training wrapper with progress reporting",
        usage="%(prog)s --progress-file FILE --train-script PATH -- [train_standalone.py args...]",
    )
    parser.add_argument("--progress-file", required=True, help="Path to write progress JSON")
    parser.add_argument(
        "--train-script",
        default="train_standalone.py",
        help="Path to train_standalone.py (default: cwd)",
    )

    args, train_args = parser.parse_known_args()

    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    exit_code = run_training(args.progress_file, args.train_script, train_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
