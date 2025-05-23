# DICOM Text SR Writer Operator

This operator writes DICOM Structured Report (SR) objects containing text results for medical imaging workflows.

## Overview

The `DICOMTextSRWriterOperator` encodes textual results into DICOM-compliant Structured Report objects, enabling standardized storage and interoperability in clinical systems.

## Requirements

- Holoscan SDK Python package
- pydicom

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator

fragment = Fragment()
sr_writer_op = DICOMTextSRWriterOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
