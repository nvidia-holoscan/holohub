# DICOM Segmentation Writer Operator

This operator writes segmentation results into DICOM Segmentation objects for medical imaging workflows.

## Overview

The `DICOMSegWriterOperator` takes segmentation data and encodes it into DICOM-compliant segmentation objects, enabling interoperability and storage in clinical systems.

## Requirements

- Holoscan SDK Python package
- pydicom

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.dicom_seg_writer_operator import DICOMSegWriterOperator

fragment = Fragment()
seg_writer_op = DICOMSegWriterOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
