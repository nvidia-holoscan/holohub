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

import cupy as cp


class Persistence:
    """Simple infinite impulse response (IIR) temporal filter."""

    def __init__(self) -> None:
        self.state: cp.ndarray | None = None

    def __call__(self, image: cp.ndarray, alpha: float = 0.5) -> cp.ndarray:
        if self.state is None or self.state.shape != image.shape:
            self.state = cp.array(image, dtype=cp.float32)

        self.state = (1.0 - alpha) * self.state + alpha * image
        return self.state


class TemporalSVD:
    """Sliding window temporal SVD denoising."""

    def __init__(self) -> None:
        self.buffer: cp.ndarray | None = None  # Shape: (features, time)

    def __call__(self, image: cp.ndarray, history: int = 5, rank: int = 3) -> cp.ndarray:
        data = cp.asarray(image, dtype=cp.float32).flatten()
        k = int(history)

        # Initialize or reset buffer if size changes
        if self.buffer is None or self.buffer.shape[0] != data.size or self.buffer.shape[1] != k:
            self.buffer = cp.tile(data[:, None], (1, k))

        # Roll buffer and update latest frame
        self.buffer = cp.roll(self.buffer, -1, axis=1)
        self.buffer[:, -1] = data

        # SVD on (Features, Time) matrix
        # U: (F, T), s: (T,), Vt: (T, T)
        U, s, Vt = cp.linalg.svd(self.buffer, full_matrices=False)

        # Keep top 'rank' singular values
        r = min(int(rank), s.size)
        s[r:] = 0.0

        # Reconstruct only the current (last) frame: U * S * Vt[:, -1]
        reconst = cp.dot(U * s, Vt[:, -1])
        return reconst.reshape(image.shape)


__all__ = ["Persistence", "TemporalSVD"]
