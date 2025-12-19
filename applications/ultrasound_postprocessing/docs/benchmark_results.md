
## Benchmark Results

The following benchmark was run on a **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** using the `presets/benchmark.yml` preset which includes all available operators in a single pipeline.

**End-to-End Latency:** ~79.7ms (~12.5 FPS) (excluding first/last 10 frames)

| Operator | Avg (ms) | Min (ms) | Max (ms) | Count |
| :--- | :--- | :--- | :--- | :--- |
| frame_source | 0.052 | 0.044 | 0.175 | 1116 |
| gamma_compression_0 | 0.144 | 0.096 | 45.079 | 1116 |
| clahe_1 | 10.504 | 10.236 | 54.741 | 1116 |
| adaptive_gray_map_2 | 0.831 | 0.657 | 43.760 | 1116 |
| median_filter_3 | 0.327 | 0.163 | 44.077 | 1116 |
| gaussian_filter_4 | 0.220 | 0.206 | 0.781 | 1116 |
| unsharp_mask_5 | 0.238 | 0.225 | 0.608 | 1116 |
| anisotropic_diffusion_6 | 3.555 | 3.465 | 4.804 | 1116 |
| bilateral_filter_7 | 3.628 | 3.430 | 45.743 | 1116 |
| guided_filter_8 | 0.683 | 0.513 | 44.084 | 1116 |
| non_local_means_9 | 15.233 | 14.775 | 58.666 | 1116 |
| svd_denoise_10 | 39.576 | 37.174 | 180.084 | 1116 |
| persistence_11 | 0.128 | 0.116 | 0.572 | 1116 |
| temporal_svd_12 | 2.837 | 2.229 | 93.624 | 1116 |
| color_map_13 | 0.347 | 0.327 | 0.857 | 1116 |
| to_rgba | 0.170 | 0.156 | 0.348 | 1116 |
| holoviz | 0.125 | 0.089 | 2.114 | 1116 |

*Note: Most pipelines will only use a subset of these operators, resulting in significantly higher FPS. The first and last 10 samples were discarded to remove startup/shutdown jitter.*

**Investigation Note:** The high maximum latency values are driven by rare periodic spikes (occurring <1% of the time). Preliminary analysis suggests this may be related to Holoscan resource contention or system-level memory management rather than the operators themselves. An internal issue should be opened to investigate these spikes further.

