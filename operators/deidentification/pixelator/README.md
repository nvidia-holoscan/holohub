# Pixelator Operator

A Pixelation-based Deidentification Operator

## Overview

In medical and sensitive imaging workflows, pixelation is a common method for deidentification. The `PixelatorOp` is a Holoscan operator that performs pixelation-based deidentification on input images, suitable for applications such as surgical video anonymization where the camera may get out of the body and capture sensitive or protected information.

## Requirements

- [Holoscan SDK](https://docs.nvidia.com/holoscan/)
- [cupy](https://cupy.dev/)

## Example Usage

```python
from holohub.operators.deidentification.pixelator import PixelatorOp
op = PixelatorOp(block_size_h=16, block_size_w=16)
```

## Parameters

- `tensor_name`: The name of the tensor to be pixelated.
- `block_size_h`: Height of the pixelation block.
- `block_size_w`: Width of the pixelation block.
