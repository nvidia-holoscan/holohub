# DICOM Data Loader Operator

This operator loads DICOM studies into memory from a folder of DICOM instance files.

## Overview

The `DICOMDataLoaderOperator` loads DICOM studies from a specified folder, making them available as a list of `DICOMStudy` objects for downstream processing in Holoscan medical imaging pipelines.

## Requirements

- Holoscan SDK Python package
- pydicom

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.dicom_data_loader_operator import DICOMDataLoaderOperator

fragment = Fragment()
dicom_loader = DICOMDataLoaderOperator(fragment, input_folder="/path/to/dicom/files")
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
