# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 5: Live GSplat Render Viewer via HoloViz

Renders a trained Gaussian Splatting model frame-by-frame and streams the
result directly to HoloViz — no intermediate disk I/O. Optionally shows
ground-truth images side-by-side for comparison.

Usage (inside Docker with display forwarding):
    python render_viewer.py \
        --data-dir  /path/to/endonerf_dataset \
        --checkpoint /path/to/fine_best_psnr.pt \
        [--gt-dir /path/to/endonerf/images] \
        [--headless] [--save-frames /path/to/output]
"""

from __future__ import annotations

import glob
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import time

import cupy as cp
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator

SCRIPT_DIR = Path(__file__).parent.resolve()


def _load_endonerf_poses(data_dir: str) -> list[dict]:
    poses_arr = np.load(os.path.join(data_dir, "poses_bounds.npy"))
    poses_raw = poses_arr[:, :-2].reshape([-1, 3, 5])
    bds = poses_arr[:, -2:]
    N = poses_raw.shape[0]
    frames = []
    for i in range(N):
        pose = poses_raw[i]
        R, T, hwf = pose[:3, :3], pose[:3, 3], pose[:3, 4]
        H, W, focal = hwf
        K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32)
        frames.append({
            "R": R.astype(np.float32), "T": T.astype(np.float32), "K": K,
            "H": int(H), "W": int(W),
            "time": i / max(N - 1, 1),
            "near": float(bds[i, 0]), "far": float(bds[i, 1]),
        })
    return frames


def _load_checkpoint(ckpt_path: str):
    from argparse import Namespace as NS

    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    splats = ckpt["splats"]
    colors = torch.cat([splats["sh0"].cuda(), splats["shN"].cuda()], dim=-2)
    result = {"colors": colors}
    has_deform = "deform_net" in ckpt and ckpt["deform_net"] is not None

    if has_deform:
        result["mode"] = "dynamic"
        result["means_base"] = splats["means"].cuda()
        result["scales_base"] = splats["scales"].cuda()
        result["quats_base"] = F.normalize(splats["quats"], p=2, dim=-1).cuda()
        result["opacities_base"] = splats["opacities"].cuda()

        from scene.deformation import deform_network
        cfg = ckpt.get("config", NS(
            bounds=1.5, kplanes_config={"grid_dimensions": 2, "input_coordinate_dim": 4,
                                         "output_coordinate_dim": 64, "resolution": [64, 64, 64, 100]},
            multires=[1, 2, 4, 8], no_grid=False, no_dx=False, no_ds=False, no_dr=False, no_do=False,
            net_width=32, timebase_pe=6, defor_depth=0, posebase_pe=10,
            scale_rotation_pe=10, opacity_pe=10, timenet_width=64, timenet_output=32,
        ))
        dnet = deform_network(cfg).cuda()
        dnet.load_state_dict(ckpt["deform_net"])
        dnet.eval()
        result["deform_net"] = dnet
    else:
        result["mode"] = "static"
        result["means"] = splats["means"].cuda()
        result["scales"] = torch.exp(splats["scales"]).cuda()
        result["quats"] = F.normalize(splats["quats"], p=2, dim=-1).cuda()
        result["opacities"] = torch.sigmoid(splats["opacities"]).cuda()

    print(f"[Renderer] Loaded {splats['means'].shape[0]} gaussians, mode={result['mode']}")
    return result


def _apply_deformation(ckpt, t):
    N = ckpt["means_base"].shape[0]
    tv = torch.tensor([[t]], device="cuda").repeat(N, 1)
    with torch.no_grad():
        md, sd, qd, od = ckpt["deform_net"](
            point=ckpt["means_base"], scales=ckpt["scales_base"],
            rotations=ckpt["quats_base"], opacity=ckpt["opacities_base"].unsqueeze(-1),
            times_sel=tv,
        )
    return {"means": md, "scales": torch.exp(sd), "quats": qd,
            "opacities": torch.sigmoid(od.squeeze(-1)), "colors": ckpt["colors"]}


def _render_frame(params, R, T, K, W, H, near, far):
    Rt = torch.tensor(R, device="cuda", dtype=torch.float32)
    Tt = torch.tensor(T, device="cuda", dtype=torch.float32)
    Kt = torch.tensor(K, device="cuda", dtype=torch.float32)
    vm = torch.eye(4, device="cuda")
    vm[:3, :3], vm[:3, 3] = Rt, Tt
    vm = vm.inverse().unsqueeze(0)
    sh_deg = int(math.sqrt(params["colors"].shape[-2]) - 1)
    rgb, _, _ = rasterization(
        means=params["means"], quats=params["quats"], scales=params["scales"],
        opacities=params["opacities"], colors=params["colors"],
        viewmats=vm, Ks=Kt.unsqueeze(0), width=W, height=H,
        sh_degree=sh_deg, near_plane=near, far_plane=far,
        render_mode="RGB", packed=False, sparse_grad=False, rasterize_mode="classic",
    )
    return torch.clamp(rgb[0, ..., :3] * 255, 0, 255).to(torch.uint8).cpu().numpy()


class GsplatRenderOp(Operator):
    """Renders GSplat frames and loops forward/backward until the window closes.

    First pass: render each frame via gsplat and cache the RGBA composite.
    Subsequent passes: replay from cache at target FPS (no GPU rendering).
    """

    def __init__(self, fragment, ckpt_data, poses, gt_files=None,
                 save_dir=None, fps=30, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self._ckpt = ckpt_data
        self._poses = poses
        self._gt_files = gt_files
        self._save_dir = save_dir
        self._fps = fps
        self._spf = 1.0 / fps
        self._cache: list[cp.ndarray] = []
        self._playback_seq: list[int] = []
        self._play_idx = 0
        self._first_pass_done = False
        self._last_emit_time = 0.0

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def _render_and_composite(self, idx: int) -> cp.ndarray:
        """Render frame *idx*, composite with GT, return CuPy RGBA float32."""
        frame = self._poses[idx]

        if self._ckpt["mode"] == "dynamic":
            params = _apply_deformation(self._ckpt, frame["time"])
        else:
            params = {k: self._ckpt[k] for k in ("means", "scales", "quats", "opacities", "colors")}

        rendered = _render_frame(
            params, frame["R"], frame["T"], frame["K"],
            frame["W"], frame["H"], frame["near"], frame["far"],
        )

        if self._save_dir and not self._first_pass_done:
            from PIL import Image
            Image.fromarray(rendered).save(
                os.path.join(self._save_dir, f"frame_{idx:05d}.png"))

        if self._gt_files and idx < len(self._gt_files):
            gt_bgr = cv2.imread(self._gt_files[idx])
            if gt_bgr is not None:
                gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
                if gt_rgb.shape[:2] != rendered.shape[:2]:
                    gt_rgb = cv2.resize(gt_rgb, (rendered.shape[1], rendered.shape[0]))
                composite = np.concatenate([gt_rgb, rendered], axis=1)
            else:
                composite = rendered
        else:
            composite = rendered

        rgba = np.ones((*composite.shape[:2], 4), dtype=np.float32)
        rgba[..., :3] = composite.astype(np.float32) / 255.0
        return cp.asarray(rgba)

    def compute(self, op_input, op_output, context):
        N = len(self._poses)

        if not self._first_pass_done:
            # First pass: render and cache
            idx = len(self._cache)
            if idx < N:
                rgba_gpu = self._render_and_composite(idx)
                self._cache.append(rgba_gpu)
                op_output.emit({"composite": rgba_gpu}, "out")
                if idx % 10 == 0 or idx == N - 1:
                    print(f"[Renderer] Rendering {idx + 1}/{N}")
                if idx == N - 1:
                    self._first_pass_done = True
                    self._playback_seq = list(range(N)) + list(range(N - 2, 0, -1))
                    self._play_idx = 0
                    print(f"[Renderer] All {N} frames cached. Looping at {self._fps} fps (close window to stop).")
                return

        # Subsequent passes: replay from cache in forward-backward loop at target FPS
        now = time.monotonic()
        elapsed = now - self._last_emit_time
        if elapsed < self._spf:
            time.sleep(self._spf - elapsed)

        frame_idx = self._playback_seq[self._play_idx]
        op_output.emit({"composite": self._cache[frame_idx]}, "out")
        self._last_emit_time = time.monotonic()
        self._play_idx = (self._play_idx + 1) % len(self._playback_seq)


class RenderViewerApp(Application):
    def __init__(self, args):
        super().__init__()
        self._args = args

    def compose(self):
        args = self._args

        # Add training code to path for deformation network imports
        training_dir = os.environ.get("TRAINING_DIR", "")
        if training_dir and training_dir not in sys.path:
            sys.path.insert(0, training_dir)

        poses = _load_endonerf_poses(args.data_dir)
        ckpt_data = _load_checkpoint(args.checkpoint)

        gt_files = None
        gt_dir = args.gt_dir or os.path.join(args.data_dir, "images")
        if os.path.isdir(gt_dir):
            gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
            if gt_files:
                print(f"[Viewer] GT images: {len(gt_files)} from {gt_dir}")
            else:
                gt_files = None

        save_dir = None
        if args.save_frames:
            save_dir = args.save_frames
            os.makedirs(save_dir, exist_ok=True)

        has_gt = gt_files is not None
        panel_count = 2 if has_gt else 1
        label = "GT | Rendered" if has_gt else "Rendered"
        print(f"[Viewer] {len(poses)} frames, layout: {label}")

        source = GsplatRenderOp(
            self, ckpt_data=ckpt_data, poses=poses, gt_files=gt_files,
            save_dir=save_dir, fps=args.fps, name="renderer",
        )

        H, W = poses[0]["H"], poses[0]["W"]
        win_w = min(W * panel_count, 1920)
        win_h = int(win_w * H / (W * panel_count))

        pool = UnboundedAllocator(self, name="pool")
        holoviz = HolovizOp(
            self, allocator=pool, name="holoviz",
            headless=args.headless,
            width=win_w, height=win_h,
            tensors=[dict(name="composite", type="color")],
            window_title=f"G-SHARP: {label}",
        )

        self.add_flow(source, holoviz, {("out", "receivers")})


def main():
    parser = ArgumentParser(description="Live GSplat Render Viewer via HoloViz")
    parser.add_argument("--data-dir", required=True,
                        help="EndoNeRF dataset directory (with poses_bounds.npy)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to GSplat .pt checkpoint")
    parser.add_argument("--gt-dir", default=None,
                        help="Ground-truth images directory (default: data-dir/images)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target playback FPS for cached loop (default: 30)")
    parser.add_argument("--save-frames", default=None,
                        help="Optionally also save rendered frames to this directory")
    args = parser.parse_args()

    app = RenderViewerApp(args)
    app.run()


if __name__ == "__main__":
    main()
