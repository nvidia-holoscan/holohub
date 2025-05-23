# MONAI Bundle Inference Operator

This operator performs inference using MONAI Bundles for medical imaging tasks.

## Overview

The `MonaiBundleInferenceOperator` loads a MONAI Bundle model and applies it to input medical images for inference, supporting flexible deployment in Holoscan pipelines.

## Requirements

- Holoscan SDK Python package
- MONAI
- torch

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.monai_bundle_inference_operator import MonaiBundleInferenceOperator

fragment = Fragment()
bundle_op = MonaiBundleInferenceOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
