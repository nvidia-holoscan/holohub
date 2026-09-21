# MONAI Bundle Inference Operator

This operator performs inference using MONAI Bundles for medical imaging tasks.

## Overview

The `MonaiBundleInferenceOperator` loads a MONAI Bundle model and applies it to input medical images for inference, supporting flexible deployment in Holoscan pipelines.

## Requirements

- Holoscan SDK Python package
- MONAI
- torch

## Disk I/O

Disk inputs and outputs support NumPy arrays in pickle-free NPY format. The
configured file name is preserved, so a `.npy` extension is optional. Pickle files,
object arrays, and NPZ archives are rejected. Use in-memory ports for `Image`,
dictionaries, and other Python objects.

Existing numeric pickle files must be converted to NPY in a trusted environment
before use. Only convert files whose source and contents you trust; this operator
does not unpickle legacy data.

## Example Usage

```python
from pathlib import Path
from holoscan.core import Fragment
from operators.medical_imaging.monai_bundle_inference_operator import MonaiBundleInferenceOperator
from operators.medical_imaging.core import AppContext, IOMapping, IOType, Image

fragment = Fragment()
app_context = AppContext({})  # Initialize with empty args dict

bundle_op = MonaiBundleInferenceOperator(
    fragment,
    name="monai_bundle",  # Optional operator name
    app_context=app_context,
    input_mapping=[
        IOMapping(
            label="image",
            data_type=Image,
            storage_type=IOType.IN_MEMORY
        )
    ],
    output_mapping=[
        IOMapping(
            label="pred",
            data_type=Image,
            storage_type=IOType.IN_MEMORY
        )
    ],
    model_name="model",  # Name of the model in the bundle
    bundle_path=Path("model/model.ts")  # Path to the MONAI bundle
)
```
