# G-SHARP Assets (Model Checkpoints)

Place model checkpoint files in this directory before running the pipeline.
The Dockerfile does **not** bake these into the image — they are mounted at
runtime via `-v /path/to/assets:/workspace/assets`.

## Required Files

```text
assets/
├── da2/
│   └── depth_anything_v2_vits.pth          # ~95 MB
└── medsam3/
    └── checkpoint_8_new_best.pt            # ~9.4 GB
```

### Depth Anything V2 (`da2/`)

Download the ViT-S checkpoint from the
[Depth-Anything-V2 repository](https://github.com/DepthAnything/Depth-Anything-V2):

```bash
wget -O assets/da2/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

### MedSAM3 (`medsam3/`)

Download the MedSAM3 checkpoint from [ChongCong/Medical-SAM3 on Hugging Face](https://huggingface.co/ChongCong/Medical-SAM3) (or use your own fine-tuned `.pt`). Place it at:

```text
assets/medsam3/checkpoint_8_new_best.pt
```

If the Hugging Face repo uses a different filename, copy or symlink that file to `checkpoint_8_new_best.pt`.

## Automatically Managed Weights

The following weights are handled automatically and do **not** need to be
placed in this directory:

| Model | Size | How It's Loaded |
| ----- | ---- | --------------- |
| VGGT-1B | ~4 GB | Auto-downloaded from HuggingFace Hub on first run; cached at `~/.cache/huggingface/` |
| VGG-16 (LPIPS) | ~528 MB | Pre-cached in the Docker image at build time |

## HuggingFace Cache

To avoid re-downloading VGGT weights on every container run, mount your local
HuggingFace cache:

```bash
-v ~/.cache/huggingface:/root/.cache/huggingface
```
