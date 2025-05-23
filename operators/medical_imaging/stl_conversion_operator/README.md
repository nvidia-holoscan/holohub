# STL Conversion Operator

This operator converts medical imaging data to STL format for 3D visualization and printing.

## Overview

The `STLConversionOperator` takes volumetric or surface data and outputs STL files, supporting workflows for 3D modeling and printing in medical imaging.

## Requirements

- Holoscan SDK Python package
- numpy-stl

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.stl_conversion_operator import STLConversionOperator

fragment = Fragment()
stl_op = STLConversionOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
