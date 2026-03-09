# Depth Anything V2 (custom fork)

This directory contains a **custom fork** of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and DINOv2-style layers used by the G-SHARP pipeline for metric depth estimation. It is not a drop-in of the upstream repository.

**Why it is in-tree:** Our version uses a sigmoid head and bounded metric depth, custom normalization, and checkpoints trained for this modified architecture. Upstream code and weights are not compatible. See **[DEV_NOTES.md](../../DEV_NOTES.md)** in the application root for the full rationale (metric depth head, normalization, checkpoint compatibility).

**License:** The code in this directory is derived from Depth Anything V2 / DINOv2. Original copyright (c) Meta Platforms, Inc. and affiliates. It is licensed under the **Apache License, Version 2.0**. See **[LICENSE](LICENSE)** in this directory.

**Checkpoint:** Use the application’s `assets/da2/` checkpoint (e.g. `depth_anything_v2_vits.pth`); it is not compatible with upstream DA2.
