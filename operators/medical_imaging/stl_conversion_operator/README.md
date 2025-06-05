# STL Conversion Operator

This operator converts medical imaging data to STL format for 3D visualization and printing.

## Overview

The `STLConversionOperator` takes volumetric or surface data and outputs STL files, supporting workflows for 3D modeling and printing in medical imaging.

## Requirements

- Holoscan SDK Python package
- numpy
- numpy-stl

## Example Usage

Here's a basic example of how to use the STLConversionOperator:

```python
from holoscan.core import Fragment
from operators.medical_imaging.stl_conversion_operator import STLConversionOperator
from pathlib import Path

# Create a fragment
fragment = Fragment()

# Initialize the STL conversion operator
stl_operator = STLConversionOperator(
    fragment,
    output_file="output/surface_mesh.stl",  # Path to save the STL file
    is_smooth=True,  # Enable mesh smoothing
    keep_largest_connected_component=True  # Keep only the largest connected component
)

# Setup the operator
stl_operator.setup()

# Example: Convert an image to STL
# Assuming 'image' is an Image object with volumetric data
stl_bytes = stl_operator._convert(image, Path("output/surface_mesh.stl"))
```

For a complete workflow example that includes loading DICOM data and converting it to STL, please refer to the tutorial on [Processing DICOM to USD with MONAI Deploy and Holoscan](../../../tutorials/dicom_to_usd_with_monai_and_holoscan/tutorial.py).

## Parameters

The STLConversionOperator accepts the following parameters:

- `output_file` (Path or str): Path where the STL file will be saved
- `class_id` (array, optional): Class label IDs to include in the conversion
- `is_smooth` (bool, optional): Whether to apply mesh smoothing (default: True)
- `keep_largest_connected_component` (bool, optional): Whether to keep only the largest connected component (default: True)
