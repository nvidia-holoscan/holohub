# DICOM Data Loader Operator

This operator loads DICOM studies into memory from a folder of DICOM instance files.

## Overview

The `DICOMDataLoaderOperator` loads DICOM studies from a specified folder, making them available as a list of `DICOMStudy` objects for downstream processing in Holoscan medical imaging pipelines.

## Requirements

- Holoscan SDK Python package
- pydicom

## Example Usage

```python
from pathlib import Path
from holoscan.core import Fragment
from operators.medical_imaging.dicom_data_loader_operator import DICOMDataLoaderOperator

fragment = Fragment()
dicom_loader = DICOMDataLoaderOperator(
    fragment,
    name="dicom_loader",  # Optional operator name
    input_folder=Path("input"),  # Path to folder containing DICOM files
    output_name="dicom_study_list",  # Name of the output port
    must_load=True  # Whether to raise an error if no DICOM files are found
)
```
