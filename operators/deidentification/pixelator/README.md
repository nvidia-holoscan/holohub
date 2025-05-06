# Pixelator: A Pixelation-based Deidentification Operator

This directory contains the `PixelatorOp` for use with NVIDIA Holoscan workflows. The operator performs pixelation-based deidentification on input images, suitable for applications such as surgical video anonymization.

## Structure

- `pixelator.py`: The main operator implementation (class `PixelatorOp`).
- `README.md`: This file.
- `metadata.json`: Operator metadata.

## Requirements
- [Holoscan SDK](https://docs.nvidia.com/holoscan/)
- [cupy](https://cupy.dev/)

## Example Usage
```python
from holohub.operators.deidentification.pixelator import PixelatorOp
op = PixelatorOp(block_size_h=16, block_size_w=16)
```

## Parameters

- `block_size_h`: Height of the pixelation block.
- `block_size_w`: Width of the pixelation block.
