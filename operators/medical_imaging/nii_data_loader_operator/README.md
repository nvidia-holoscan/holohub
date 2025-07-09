# NIfTI Data Loader Operator

This operator loads NIfTI (nii/nii.gz) medical images for use in Holoscan pipelines.

## Overview

The `NIIDataLoaderOperator` reads NIfTI files and makes them available as in-memory images for downstream processing in medical imaging workflows.

## Requirements

- Holoscan SDK Python package (version >= 1.0.3)
- SimpleITK
- numpy

## Example Usage

```python
from pathlib import Path
from holoscan.core import Fragment
from operators.medical_imaging.nii_data_loader_operator import NiftiDataLoader

# Create a fragment
fragment = Fragment()

# Initialize the NIfTI loader with a path to your NIfTI file
nii_path = Path("path/to/your/image.nii")  # or .nii.gz
nii_loader = NiftiDataLoader(fragment, input_path=nii_path)

# The operator can be used in a pipeline
# The output port 'image' will contain the loaded image as a numpy array
# You can connect it to other operators that expect image data
```

## Input/Output Ports

- Input:
  - `image_path` (optional): Path to the NIfTI file. If not provided, uses the path specified during initialization.
- Output:
  - `image`: Numpy array containing the loaded image data.

## Notes

- The operator supports both .nii and .nii.gz file formats
- The output image is transposed to match the expected orientation (axes order: [2, 1, 0])
