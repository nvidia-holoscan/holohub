# DICOM Series Selector Operator

This operator selects specific DICOM series from a set of studies for further processing in medical imaging workflows.

## Overview

The `DICOMSeriesSelectorOperator` enables filtering and selection of relevant DICOM series, streamlining downstream analysis and processing in Holoscan pipelines.

## Requirements

- Holoscan SDK Python package
- pydicom

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.dicom_series_selector_operator import DICOMSeriesSelectorOperator

fragment = Fragment()
selector_op = DICOMSeriesSelectorOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
