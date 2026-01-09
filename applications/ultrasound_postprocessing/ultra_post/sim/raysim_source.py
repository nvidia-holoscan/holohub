# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - GPU runtime dependency
    import cupy as cp
except ImportError as exc:  # pragma: no cover - handled lazily at runtime
    raise ImportError(
        "CuPy is required to run the simulated pipeline (`pip install cupy-cuda12x`)."
    ) from exc

try:  # pragma: no cover - optional dependency for simulation
    import raysim.cuda as rs
except ImportError as exc:  # pragma: no cover - handled lazily at runtime
    raise ImportError(
        "The `raysim` package from i4h-sensor-simulation is required. "
        "Install it via `pip install -e external/i4h-sensor-simulation/ultrasound-raytracing`."
    ) from exc

Tensor = cp.ndarray


@dataclass
class RaysimSweepConfig:
    """Configuration controlling the synthetic sweep."""

    frames_per_loop: int = 60
    sweep_range_mm: Tuple[float, float] = (-20.0, 20.0)
    probe_rotation_deg: Tuple[float, float, float] = (0.0, 180.0, 0.0)
    probe_height_mm: float = 0.0
    probe_depth_mm: float = 0.0
    probe_type: str = "curvilinear"
    num_elements_x: int = 256
    sector_angle_deg: float = 73.0
    radius_mm: float = 45.0
    frequency_mhz: float = 5.0
    elevational_height_mm: float = 7.0
    num_elevational_samples: int = 10
    dynamic_range_db: Tuple[float, float] = (-60.0, 0.0)
    b_mode_size: Tuple[int, int] = (512, 512)
    conv_psf: bool = True
    buffer_size: int = 4096
    t_far_mm: float = 180.0
    enable_cuda_timing: bool = False
    median_clip_filter: bool = False
    world_name: str = "fat"
    sphere_depths_mm: Sequence[float] = field(default_factory=lambda: (-145.0, -225.0))
    sphere_radius_mm: float = 40.0
    sphere_material: str = "water"

    def positions_mm(self) -> np.ndarray:
        start, end = self.sweep_range_mm
        return np.linspace(start, end, num=self.frames_per_loop, dtype=np.float32)

    def rotation_radians(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.probe_rotation_deg, dtype=np.float32))


class RaysimFrameGenerator:
    """Convenience bridge that emits synthetic B-mode frames from raysim."""

    def __init__(self, config: Optional[RaysimSweepConfig] = None) -> None:
        self.config = config or RaysimSweepConfig()
        self._materials = rs.Materials()
        self._world = rs.World(self.config.world_name)
        self._build_default_scene()

        self._simulator = rs.RaytracingUltrasoundSimulator(self._world, self._materials)
        self._sim_params = self._create_sim_params()
        self._positions = self.config.positions_mm()
        self._frame_index = 0
        self._meta: Dict[str, object] = {}

    def _build_default_scene(self) -> None:
        material_idx = self._materials.get_index(self.config.sphere_material)
        for depth in self.config.sphere_depths_mm:
            center = np.array([0.0, 0.0, float(depth)], dtype=np.float32)
            sphere = rs.Sphere(center, float(self.config.sphere_radius_mm), material_idx)
            self._world.add(sphere)

    def _create_sim_params(self) -> rs.SimParams:
        params = rs.SimParams()
        params.conv_psf = bool(self.config.conv_psf)
        params.buffer_size = int(self.config.buffer_size)
        params.t_far = float(self.config.t_far_mm)
        params.enable_cuda_timing = bool(self.config.enable_cuda_timing)
        params.median_clip_filter = bool(self.config.median_clip_filter)
        params.b_mode_size = tuple(int(dim) for dim in self.config.b_mode_size)
        return params

    def _next_pose(self, x_pos_mm: float) -> rs.Pose:
        position = np.array(
            [x_pos_mm, float(self.config.probe_height_mm), float(self.config.probe_depth_mm)],
            dtype=np.float32,
        )
        rotation = self.config.rotation_radians()
        return rs.Pose(position=position, rotation=rotation)

    def _make_probe(self, pose: rs.Pose) -> rs.CurvilinearProbe:
        if self.config.probe_type != "curvilinear":
            raise NotImplementedError("Only curvilinear probes are supported in this demo bridge.")
        return rs.CurvilinearProbe(
            pose,
            num_elements_x=int(self.config.num_elements_x),
            sector_angle=float(self.config.sector_angle_deg),
            radius=float(self.config.radius_mm),
            frequency=float(self.config.frequency_mhz),
            elevational_height=float(self.config.elevational_height_mm),
            num_el_samples=int(self.config.num_elevational_samples),
        )

    def next_frame(self) -> Tensor:
        if self._positions.size == 0:
            raise RuntimeError("Sweep configuration must yield at least one position.")
        position = float(self._positions[self._frame_index % self._positions.size])
        self._frame_index += 1

        pose = self._next_pose(position)
        probe = self._make_probe(pose)
        raw_image = self._simulator.simulate(probe, self._sim_params)

        tensor = self._normalize(cp.asarray(raw_image, dtype=cp.float32))
        self._meta = {
            "position_mm": position,
            "frame_index": self._frame_index,
            "b_mode_shape": tuple(int(dim) for dim in tensor.shape),
        }
        return tensor

    def _normalize(self, tensor: Tensor) -> Tensor:
        lo, hi = self.config.dynamic_range_db
        if hi <= lo:
            hi = lo + 1e-3
        scaled = (tensor - float(lo)) / float(hi - lo)
        return cp.clip(scaled, 0.0, 1.0)

    @property
    def metadata(self) -> Dict[str, object]:
        """Metadata for the most recent frame."""

        return dict(self._meta)

    def reset(self) -> None:
        self._frame_index = 0
        self._meta = {}
