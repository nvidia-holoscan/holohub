# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from . import adaptive_gray, anisotropic, bilateral, clahe, colormap, std, gamma, guided, matte, nlm, svd, temporal

FILTERS = {
    "adaptive_gray_map": adaptive_gray.adaptive_gray_map,
    "anisotropic_diffusion": anisotropic.anisotropic_diffusion,
    "auto_matte": matte.AutoMatte,
    "bilateral_filter": bilateral.bilateral_filter,
    "clahe": clahe.clahe,
    "color_map": colormap.color_map,
    "median_filter": std.median_filter,
    "gaussian_filter": std.gaussian_filter,
    "unsharp_mask": std.unsharp_mask,
    "gamma_compression": gamma.gamma_compression,
    "guided_filter": guided.guided_filter,
    "non_local_means": nlm.non_local_means,
    "svd_denoise": svd.svd_denoise,
    "persistence": temporal.Persistence,
    "temporal_svd": temporal.TemporalSVD,
}

DEFAULT_PARAMS = {
    "adaptive_gray_map": {"radius": 5, "beta": 1.0, "ref_level": 0.5, "auto_ref": True, "preserve_mean": True},
    "anisotropic_diffusion": {"num_iter": 10, "kappa": 0.1, "gamma": 0.1, "function": "exponential"},
    "auto_matte": {"filters": []},
    "bilateral_filter": {"radius": 3, "spatial_sigma": 2.0, "range_sigma": 0.1, "per_channel": True},
    "clahe": {"tiles": (8, 8), "clip_limit": 0.01, "nbins": 256},
    "color_map": {"mode": colormap.MapChoice.blue, "blend": 1.0, "enable": True},
    "median_filter": {"size": 3},
    "gaussian_filter": {"sigma": 1.0},
    "unsharp_mask": {"sigma": 1.0, "amount": 1.5, "threshold": 0.0},
    "gamma_compression": {"gamma": 1.0},
    "guided_filter": {"radius": 5, "eps": 0.01, "use_luminance": True},
    "non_local_means": {"h": 0.1, "patch_size": 7, "patch_distance": 11, "fast_mode": True},
    "svd_denoise": {"rank": 32, "suppress": "low", "shrink": 0.0},
    "persistence": {"alpha": 0.5},
    "temporal_svd": {"history": 5, "rank": 3},
}

__all__ = ["FILTERS", "DEFAULT_PARAMS"]
