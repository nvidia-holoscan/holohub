#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch

# Import TapNext modules
from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TapNextONNXWrapper(torch.nn.Module):
    """
    ONNX-exportable wrapper for TapNext that includes query point creation
    and forward tracking logic.
    """

    def __init__(self, tapnext_model):
        super().__init__()
        self.model = tapnext_model

    def forward(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        step: Optional[torch.Tensor] = None,
        rg_lru_state: Optional[torch.Tensor] = None,
        conv1d_state: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for ONNX export.

        Args:
            video: Video tensor of shape (1, T, H, W, C) normalized to [-1, 1]
            query_frame: Query frame index (default: 0)

        Returns:
            Tuple of (tracks, visibility)
            - tracks: Shape (1, T_out, N, 2) - predicted trajectories
            - visibility: Shape (1, T_out, N) - visibility predictions
        """
        if step is not None:
            if rg_lru_state is None or conv1d_state is None:
                raise ValueError(
                    "rg_lru_state and conv1d_state must be provided when step is not None"
                )

            tracking_state = {
                "step": step,
                "query_points": query_points,
                # rg_lru_state: ([12, 1124, 3, 768])
                # conv1d_state: ([12, 1124, 768])
                # They have to be converted to a list of dictionaries. torch.unbind returns a tuple of tensors.
                "hidden_state": [
                    {"rg_lru_state": state[0], "conv1d_state": state[1]}
                    for state in zip(torch.unbind(rg_lru_state, 0), torch.unbind(conv1d_state, 0))
                ],
            }
            query_points = None
        else:
            tracking_state = None

        curr_tracks, curr_track_logits, curr_visible_logits, tracking_state = self.model(
            video, query_points=query_points, state=tracking_state
        )
        # Convert from (y, x) to (x, y) format
        curr_tracks = torch.stack([curr_tracks[..., 1], curr_tracks[..., 0]], dim=-1)
        # step: int
        # query_points: torch.Tensor  # Float["*B Q 3"]
        # hidden_state: Optional[List[RecurrentBlockCache]] = None
        step = tracking_state["step"]

        # hidden_state: list(dict(rg_lru_state torch.Tensor, conv1d_state torch.Tensor))

        rg_lru_state = torch.stack(
            [state["rg_lru_state"] for state in tracking_state["hidden_state"]], dim=0
        )
        conv1d_state = torch.stack(
            [state["conv1d_state"] for state in tracking_state["hidden_state"]], dim=0
        )

        return curr_tracks, curr_track_logits, curr_visible_logits, step, rg_lru_state, conv1d_state
        # return step, rg_lru_state, conv1d_state


def disable_torch_compile_for_export():
    """Temporarily disable torch.compile decorators for ONNX export."""
    original_torch_compile = torch.compile

    def no_op_compile(func=None, *args, **kwargs):
        if func is None:
            return lambda f: f
        return func

    torch.compile = no_op_compile
    return original_torch_compile


def restore_torch_compile(original_torch_compile):
    """Restore original torch.compile function."""
    torch.compile = original_torch_compile


def export_to_onnx(
    model, video_tensor, output_path, query_points, step=None, rg_lru_state=None, conv1d_state=None
):
    """Export TapNext model to ONNX format."""
    print(f"Exporting model to {output_path}...")

    # Set model to eval mode
    model.eval()

    name = "tapnext_init.onnx" if step is None else "tapnext_forward.onnx"

    if step is None:
        # Export to ONNX
        torch.onnx.export(
            model,
            args=(),
            kwargs=dict(video=video_tensor, query_points=query_points),
            f=os.path.join(output_path, name),
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=["video", "query_points"],
            output_names=[
                "tracks",
                "track_logits",
                "visible_logits",
                "step",
                "rg_lru_state",
                "conv1d_state",
            ],
            # dynamo=True
        )
    else:
        # Export to ONNX
        torch.onnx.export(
            model,
            args=(),
            kwargs=dict(
                video=video_tensor,
                query_points=query_points,
                step=step,
                rg_lru_state=rg_lru_state,
                conv1d_state=conv1d_state,
            ),
            f=os.path.join(output_path, name),
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=["video", "query_points", "step", "rg_lru_state", "conv1d_state"],
            output_names=[
                "tracks",
                "track_logits",
                "visible_logits",
                "step",
                "rg_lru_state",
                "conv1d_state",
            ],
            # dynamo=True
        )

    print(f"Model exported to {output_path}")


def create_query_points(grid_size: int, query_frame: int, height: int, width: int) -> np.ndarray:
    """
    Create query points grid for TapNext.

    Args:
        grid_size: Grid size for point initialization
        query_frame: Query frame index
        height: Video height
        width: Video width

    Returns:
        Query points array of shape (1, N, 3) with format (frame_idx, x, y)
    """
    # Create grid similar to tapnext_scratch.py
    margin = 8  # Margin from edges
    ys, xs = np.meshgrid(
        np.linspace(margin, height - margin, grid_size),
        np.linspace(margin, width - margin, grid_size),
    )

    # TapNext expects (frame_idx, x, y) format
    query_points = np.stack(
        [
            np.full(len(xs.flatten()), query_frame, dtype=np.float32),
            xs.flatten().astype(np.float32),
            ys.flatten().astype(np.float32),
        ],
        axis=1,
    )[
        None
    ]  # Add batch dimension

    return query_points


def main():
    parser = argparse.ArgumentParser(description="Export TapNext model to ONNX and validate")
    parser.add_argument(
        "--checkpoint",
        default="bootstapnext_ckpt.npz",
        help="Path to TapNext checkpoint (JAX format)",
    )
    # parser.add_argument(
    #     "--video", required=True, help="Path to test video (numpy file)"
    # )
    parser.add_argument("--output_path", default="output/", help="Output ONNX file path")
    parser.add_argument("--grid_size", type=int, default=15, help="Grid size for query points")
    parser.add_argument("--query_frame", type=int, default=0, help="Query frame index")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Temporarily disable torch.compile for ONNX export
    original_torch_compile = disable_torch_compile_for_export()

    # Determine device
    device = args.device
    print(f"Using device: {device}")

    # Load TapNext model
    print("Loading TapNext model...")
    tapnext_model = TAPNext(
        image_size=(256, 256), n_query_points=args.grid_size * args.grid_size, device=device
    )
    tapnext_model = restore_model_from_jax_checkpoint(tapnext_model, args.checkpoint)

    # Disable torch.compile on the model
    if hasattr(tapnext_model, "_orig_mod"):
        tapnext_model = tapnext_model._orig_mod

    # Move to device and set to eval mode
    tapnext_model = tapnext_model.to(device).eval()

    query_points = create_query_points(
        args.grid_size, query_frame=args.query_frame, height=256, width=256
    )
    query_points = torch.from_numpy(query_points).to(device)

    # Create ONNX wrapper
    model = TapNextONNXWrapper(tapnext_model).to(device)

    # Create dummy video tensor for export
    video_tensor = torch.rand([1, 4, 256, 256, 3], device=device)

    print("Running PyTorch inference...")
    with torch.no_grad():
        _, _, _, step, rg_lru_state, conv1d_state = model(video_tensor[:, :1], query_points)

    # Export initial model (no state)
    export_to_onnx(model, video_tensor[:, :1], args.output_path, query_points)

    # Run initial forward pass to get state for the forward model export
    with torch.no_grad():
        _, _, _, step, rg_lru_state, conv1d_state = model(video_tensor[:, :1], query_points)

    # Export forward model (with state)
    export_to_onnx(
        model,
        video_tensor[:, 1:2],
        args.output_path,
        query_points,
        step=step,
        rg_lru_state=rg_lru_state,
        conv1d_state=conv1d_state,
    )

    print("[SUCCESS] Models exported to", args.output_path)

    restore_torch_compile(original_torch_compile)


if __name__ == "__main__":
    main()
