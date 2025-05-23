# DICOM Segmentation Writer (All Frames) Operator

This operator writes segmentation results into DICOM Segmentation objects (all frames) for medical imaging workflows.

## Overview

The `DICOMSegWriterOperatorAllFrames` encodes segmentation data into DICOM-compliant segmentation objects, supporting multi-frame output for clinical interoperability.

## Requirements

- Holoscan SDK Python package
- pydicom

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.dicom_seg_writer_operator_all_frames import DICOMSegWriterOperatorAllFrames

fragment = Fragment()
seg_writer_op = DICOMSegWriterOperatorAllFrames(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
