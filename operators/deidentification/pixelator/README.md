# Pixelator Operator

This directory contains the `PixelatorOp` for use with NVIDIA Holoscan workflows. The operator performs pixelation-based deidentification on input images, suitable for applications such as surgical video anonymization.

## Structure

- `pixelator_op.py`: The main operator implementation (class `PixelatorOp`).
- `README.md`: This file.

## Requirements
- [Holoscan SDK](https://docs.nvidia.com/holoscan/)
- [cupy](https://cupy.dev/)

## Example Usage
```python
from operators.deidentification.pixelator.pixelator_op import PixelatorOp
op = PixelatorOp(block_size_h=16, block_size_w=16)
