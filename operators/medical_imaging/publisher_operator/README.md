# Publisher Operator

This operator publishes medical imaging data to downstream consumers or external systems.

## Overview

The `PublisherOperator` enables flexible publishing of processed medical imaging data for visualization, storage, or further analysis in Holoscan pipelines.

## Requirements

- Holoscan SDK Python package

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.publisher_operator import PublisherOperator
from pathlib import Path

# Create a fragment
fragment = Fragment()

# Initialize the publisher operator with input and output folders
pub_op = PublisherOperator(
    fragment,
    input_folder=Path("path/to/input"),  # Folder containing input and segment mask files
    output_folder=Path("path/to/output")  # Folder where published files will be saved
)

# Add the operator to the fragment
fragment.add_operator(pub_op)
```

The operator expects:

- Input folder containing medical imaging files (nii, nii.gz, or mhd format)
- Output folder where the published files will be saved
- The operator will automatically find density and mask files in the input folder
- Published files will include the original images and configuration files for visualization
