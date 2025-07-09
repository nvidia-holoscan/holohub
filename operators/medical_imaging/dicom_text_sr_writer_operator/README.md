# DICOM Text SR Writer Operator

This operator writes DICOM Structured Report (SR) objects containing text results for medical imaging workflows.

## Overview

The `DICOMTextSRWriterOperator` encodes textual results into DICOM-compliant Structured Report objects, enabling standardized storage and interoperability in clinical systems.

## Requirements

- Holoscan SDK Python package
- pydicom
- highdicom

## Example Usage

```python
from pathlib import Path
from holoscan.core import Fragment
from operators.medical_imaging.dicom_text_sr_writer_operator import DICOMTextSRWriterOperator
from operators.medical_imaging.utils.dicom_utils import ModelInfo, EquipmentInfo

fragment = Fragment()
sr_writer_op = DICOMTextSRWriterOperator(
    fragment,
    name="sr_writer",  # Optional operator name
    output_folder=Path("output"),  # Path to save the generated DICOM file(s)
    model_info=ModelInfo(
        creator="ExampleCreator",
        name="ExampleModel",
        version="1.0.0",
        uid="1.2.3.4.5.6.7.8.9"
    ),
    equipment_info=EquipmentInfo(
        manufacturer="ExampleManufacturer",
        manufacturer_model="ExampleModel",
        series_number="0000",
        software_version_number="1.0.0"
    ),
    copy_tags=True,  # Set to True to copy tags from a DICOMSeries
    custom_tags={"PatientName": "DOE^JOHN"}  # Optional: custom DICOM tags as a dict
)
```
