# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Demo for DepthToPointCloudOp.

Generates a synthetic organized depth image (a gently tilting plane) plus an aligned RGB
image entirely on the GPU, deprojects it into an organized point cloud with
DepthToPointCloudOp, and validates the result. No camera or dataset is required, so the
app runs in CI. See README.md for wiring a real Intel RealSense camera or adding Holoviz
3D rendering.
"""

import argparse

import cupy as cp
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType

from holohub.depth_to_point_cloud import DepthToPointCloudOp


class SyntheticDepthGeneratorOp(Operator):
    """Emit a synthetic float32 depth image (meters) and an aligned uint8 RGB image."""

    def __init__(self, fragment, *args, width=640, height=480, **kwargs):
        self.width = width
        self.height = height
        self.frame = 0
        ys, xs = cp.meshgrid(
            cp.arange(height, dtype=cp.float32),
            cp.arange(width, dtype=cp.float32),
            indexing="ij",
        )
        self._xs = xs
        self._ys = ys
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("depth")
        spec.output("color")

    def compute(self, op_input, op_output, context):
        t = self.frame * 0.05
        # A tilted plane in meters: ~1.0 m near the top-left, increasing across the frame,
        # with a slow global oscillation so successive frames differ.
        depth = (
            1.0 + 0.5 * (self._xs / self.width) + 0.4 * (self._ys / self.height) + 0.3 * cp.sin(t)
        ).astype(cp.float32)

        r = (255.0 * self._xs / self.width).astype(cp.uint8)
        g = (255.0 * self._ys / self.height).astype(cp.uint8)
        b = cp.full_like(r, 128)
        color = cp.ascontiguousarray(cp.stack([r, g, b], axis=-1))  # HxWx3 uint8

        op_output.emit({"depth": depth}, "depth")
        op_output.emit({"color": color}, "color")
        self.frame += 1


class PointCloudStatsOp(Operator):
    """Pull the point cloud and report valid-point count and Z range (CI-friendly sink)."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        pc = cp.asarray(msg["point_cloud"])  # HxWx3 float32
        z = pc[..., 2]
        valid = ~cp.isnan(z)
        n_valid = int(valid.sum().get())

        # The colored path is connected, so a colors tensor must accompany the cloud and
        # share its H x W footprint (3 uint8 channels).
        colors = cp.asarray(msg["colors"])  # HxWx3 uint8
        if colors.shape[:2] != pc.shape[:2] or colors.shape[2] != 3:
            raise RuntimeError(
                f"colors shape {colors.shape} does not match cloud {pc.shape[:2]} x 3"
            )

        if n_valid:
            zmin = float(z[valid].min().get())
            zmax = float(z[valid].max().get())
            print(
                f"[depth_to_point_cloud_demo] points={pc.shape[0] * pc.shape[1]} "
                f"valid={n_valid} z=[{zmin:.3f}, {zmax:.3f}] m colors={tuple(colors.shape)}"
            )
        else:
            print("[depth_to_point_cloud_demo] no valid points")


class DepthToPointCloudDemoApp(Application):
    def __init__(self, frames=100, width=640, height=480):
        super().__init__()
        self._frames = frames
        self._width = width
        self._height = height

    def compose(self):
        generator = SyntheticDepthGeneratorOp(
            self,
            CountCondition(self, count=self._frames),
            name="generator",
            width=self._width,
            height=self._height,
        )

        # Two device tensors per frame (HxWx3 float32 point cloud + HxWx3 uint8 colors) drawn
        # from this pool; size each block for the larger (float32 XYZ) output and keep enough
        # blocks for both tensors plus one frame of pipelining headroom.
        out_blocks = 4
        block_size = self._width * self._height * 3 * 4  # float32 XYZ is the larger output
        cloud = DepthToPointCloudOp(
            self,
            name="point_cloud",
            allocator=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=block_size,
                num_blocks=out_blocks,
            ),
            # Pinhole intrinsics for the synthetic camera: square pixels (fx == fy) with the
            # principal point at the image center. A single focal length is used for both axes
            # by design; the focal length is independent of the image aspect ratio.
            fx=float(self._width) * 0.8,
            fy=float(self._width) * 0.8,
            cx=(self._width - 1) / 2.0,
            cy=(self._height - 1) / 2.0,
            depth_scale=1.0,  # synthetic depth is already in meters
            depth_min=0.1,
            depth_max=10.0,
            cuda_stream_pool=CudaStreamPool(self, name="stream_pool", reserved_size=4),
        )

        sink = PointCloudStatsOp(self, name="stats")

        self.add_flow(generator, cloud, {("depth", "depth")})
        self.add_flow(generator, cloud, {("color", "color")})
        self.add_flow(cloud, sink, {("point_cloud", "in")})


def main():
    parser = argparse.ArgumentParser(description="DepthToPointCloudOp synthetic demo")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    for name, value in (
        ("--frames", args.frames),
        ("--width", args.width),
        ("--height", args.height),
    ):
        if value <= 0:
            parser.error(f"{name} must be a positive integer (got {value})")

    app = DepthToPointCloudDemoApp(frames=args.frames, width=args.width, height=args.height)
    app.run()


if __name__ == "__main__":
    main()
