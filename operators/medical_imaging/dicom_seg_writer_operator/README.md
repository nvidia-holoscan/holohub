# DICOM Segmentation Writer Operator

This operator writes segmentation results into DICOM Segmentation objects for medical imaging workflows.

## Overview

The `DICOMSegmentationWriterOperator` takes segmentation data and encodes it into DICOM-compliant segmentation objects, enabling interoperability and storage in clinical systems.

## Requirements

- Holoscan SDK Python package
- pydicom
- highdicom
- SimpleITK (for image I/O, if using NIfTI or MHD files)
- numpy

## Example Usage

```python
from pathlib import Path
from holoscan.core import Fragment
from operators.medical_imaging.dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from highdicom import codes

fragment = Fragment()
seg_writer_op = DICOMSegmentationWriterOperator(
    fragment,
    name="seg_writer",  # Optional operator name
    segment_descriptions=[
        SegmentDescription(
            segment_label="Liver",
            segmented_property_category=codes.DCM.Organ,
            segmented_property_type=codes.DCM.Liver,
            algorithm_name="ExampleAlgorithm",
            algorithm_version="1.0.0"
        )
    ],
    output_folder=Path("output"),  # Path to save the generated DICOM file(s)
    custom_tags={"PatientName": "DOE^JOHN"},  # Optional: custom DICOM tags as a dict
    omit_empty_frames=True  # Whether to omit frames with no segmentation
)
```
