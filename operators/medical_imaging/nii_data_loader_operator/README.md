# NIfTI Data Loader Operator

This operator loads NIfTI (nii/nii.gz) medical images for use in Holoscan pipelines.

## Overview

The `NIIDataLoaderOperator` reads NIfTI files and makes them available as in-memory images for downstream processing in medical imaging workflows.

## Requirements

- Holoscan SDK Python package
- nibabel

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.nii_data_loader_operator import NIIDataLoaderOperator

fragment = Fragment()
nii_loader = NIIDataLoaderOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
