# DICOM Series to Volume Operator

This operator converts a DICOM series into a volumetric image for downstream analysis in medical imaging workflows.

## Overview

The `DICOMSeriesToVolumeOperator` reads a DICOM series and constructs a volume image suitable for 3D processing and visualization in Holoscan pipelines.

## Requirements

- Holoscan SDK Python package
- pydicom
- numpy

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

fragment = Fragment()
vol_op = DICOMSeriesToVolumeOperator(
    fragment,
    name="series_to_volume"  # Optional operator name
)
```
