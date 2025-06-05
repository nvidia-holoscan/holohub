# DICOM Encapsulated PDF Writer Operator

This operator writes encapsulated PDF documents into DICOM format for medical imaging workflows.

## Overview

The `DICOMEncapsulatedPDFWriterOperator` converts PDF files into DICOM-compliant encapsulated PDF objects for storage and interoperability in medical imaging pipelines.

## Requirements

- Holoscan SDK Python package
- pydicom
- PyPDF2

## Example Usage

```python
from pathlib import Path
from holoscan.core import Fragment
from operators.medical_imaging.dicom_encapsulated_pdf_writer_operator import DICOMEncapsulatedPDFWriterOperator
from operators.medical_imaging.utils.dicom_utils import ModelInfo, EquipmentInfo

fragment = Fragment()
pdf_writer_op = DICOMEncapsulatedPDFWriterOperator(
    fragment,
    name="pdf_writer",  # Optional operator name
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
