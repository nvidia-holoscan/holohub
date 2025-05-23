# Inference Operator

This operator provides a base class for running inference in medical imaging pipelines.

## Overview

The `InferenceOperator` serves as a foundation for building specialized inference operators, handling model loading, execution, and result management.

## Requirements

- Holoscan SDK Python package
- torch (optional, for deep learning models)

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.inference_operator import InferenceOperator

fragment = Fragment()
infer_op = InferenceOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
