# DICOM Encapsulated PDF Writer Operator

This operator writes encapsulated PDF documents into DICOM format for medical imaging workflows.

## Overview

The `DICOMEncapsulatedPDFWriterOperator` converts PDF files into DICOM-compliant encapsulated PDF objects for storage and interoperability in medical imaging pipelines.

## Requirements

- Holoscan SDK Python package
- pydicom

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.dicom_encapsulated_pdf_writer_operator import DICOMEncapsulatedPDFWriterOperator

fragment = Fragment()
pdf_writer_op = DICOMEncapsulatedPDFWriterOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
