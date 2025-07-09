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

class MyInferenceOperator(InferenceOperator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        
    def pre_process(self, data, *args, **kwargs):
        # Implement preprocessing logic
        return data
        
    def predict(self, data, *args, **kwargs):
        # Implement inference logic
        return data
        
    def post_process(self, data, *args, **kwargs):
        # Implement postprocessing logic
        return data

fragment = Fragment()
inference_op = MyInferenceOperator(
    fragment,
    name="my_inference"  # Optional operator name
)
```
