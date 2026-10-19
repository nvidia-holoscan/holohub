# MONAI Bundle Inference Operator

This operator performs inference using MONAI Bundles for medical imaging tasks.

## Overview

The `MonaiBundleInferenceOperator` loads a MONAI Bundle model and applies it to input medical images for inference, supporting flexible deployment in Holoscan pipelines.

## Requirements

- Holoscan SDK Python package
- MONAI
- torch

## Tests

The disk I/O regression suite runs in the consuming `imaging_ai_segmentator`
application's CTest suite. Run the focused test from the repository root without
sample dataset downloads:

```bash
./holohub test imaging_ai_segmentator --language python \
  --cmake-options="-DHOLOHUB_DOWNLOAD_DATASETS=OFF" \
  --ctest-options="-DCTEST_TEST_INCLUDE=^imaging_ai_segmentator_disk_io_test$"
```

With the operator's runtime dependencies and pytest installed, it can also run
directly from the repository root:

```bash
python -m pytest -v applications/imaging_ai_segmentator/test_monai_bundle_disk_io.py
```

The suite exercises the real MONAI compute path with CPU tensors. The CUDA output
case also runs when a CUDA-enabled PyTorch installation and GPU are available.

## Path-based I/O

Input `Path` values must refer to pickle-free NPY arrays with dtypes supported by
`torch.from_numpy`. A `.npy` extension is optional. Pickle files, object arrays,
and NPZ archives are rejected. Pass `Image`, dictionaries, and other Python
objects through in-memory ports.

Holoscan graphs use in-memory output ports; save emitted arrays with a downstream
writer. The legacy path-based output helper writes NPY only when its output
context provides a directory through `get(name)`, preserving the configured file
name. It does not add support for `IOType.DISK` output-port registration.

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
