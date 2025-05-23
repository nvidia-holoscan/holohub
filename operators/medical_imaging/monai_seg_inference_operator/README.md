# MONAI Segmentation Inference Operator

This segmentation operator uses MONAI transforms and Sliding Window Inference to segment medical images.

## Overview

The `MonaiSegInferenceOperator` performs pre-transforms on input images, runs segmentation inference using a specified model, and applies post-transforms. The segmentation result is returned as an in-memory image object and can optionally be saved to disk.

## Requirements

- Holoscan SDK Python package
- MONAI
- torch

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.monai_seg_inference_operator import MonaiSegInferenceOperator

fragment = Fragment()
seg_op = MonaiSegInferenceOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
