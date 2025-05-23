# PNG Converter Operator

This operator converts medical images to PNG format for visualization or storage.

## Overview

The `PNGConverterOperator` takes medical imaging data and outputs PNG images, facilitating integration with visualization tools and pipelines.

## Requirements

- Holoscan SDK Python package
- Pillow

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.png_converter_operator import PNGConverterOperator

fragment = Fragment()
png_op = PNGConverterOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
