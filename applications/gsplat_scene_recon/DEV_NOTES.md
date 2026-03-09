# G-SHARP Development Notes

## Depth Anything V2 — Why We Cannot Use Upstream

**Date:** 2026-03-05
**Status:** Custom fork required; upstream is not compatible.

### Background

The Depth Anything V2 (DA2) model code under this application was initially
originally placed in `third_party/depth_anything_v2/`, suggesting it was a vanilla copy
of the upstream repository ([DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)).
An audit on 2026-03-05 revealed this is **not** the case — the code contains
custom modifications that are incompatible with the upstream architecture.

### Modifications vs. Upstream (`dpt.py`)

1. **Metric depth output head** — The constructor accepts a `max_depth`
   parameter (default 20.0). The forward pass computes
   `depth = sigmoid(head(features)) * max_depth`, producing bounded metric
   depth in `[0, max_depth]`. Upstream uses `depth = relu(head(features))`,
   which produces unbounded relative depth.

2. **Sigmoid activation** — The DPT output convolution ends with
   `nn.Sigmoid()` instead of upstream's `nn.ReLU(True), nn.Identity()`.
   This changes the weight space: our checkpoint's final-layer weights are
   trained for sigmoid output and will produce incorrect values if loaded
   into the upstream ReLU architecture.

3. **Custom normalization constants** — Input preprocessing uses
   `mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]` (uniform grayscale
   normalization), replacing the standard ImageNet
   `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`. This indicates
   the model was fine-tuned with this normalization; using upstream norms
   would degrade accuracy.

### Consequence

The checkpoint `assets/da2/depth_anything_v2_vits.pth` was trained for this
modified architecture. It **cannot** be loaded into the upstream DA2 code —
the layer structure (Sigmoid vs ReLU) and normalization differ. Conversely,
upstream pre-trained weights cannot be used with our modified code without
retraining.

### Decision

The DA2 model code remains bundled with the application as custom NVIDIA work
derived from Depth Anything V2 (Apache-2.0 for the Small variant). It should
not be labeled or treated as unmodified third-party code.
