# PNG Converter Operator

This operator converts medical images to PNG format for visualization or storage.

## Overview

The `PNGConverterOperator` takes medical imaging data and outputs PNG images, facilitating integration with visualization tools and pipelines.

## Requirements

- Holoscan SDK Python package
- Pillow

## Example Usage

```python
from pathlib import Path
from holoscan.core import Fragment
from operators.medical_imaging.png_converter_operator import PNGConverterOperator
from operators.medical_imaging.core import Image
import numpy as np

# Create a Fragment
fragment = Fragment()

# Create output directory
output_folder = Path("output_png")
output_folder.mkdir(exist_ok=True)

# Create the PNG converter operator
png_op = PNGConverterOperator(
    fragment,
    output_folder=output_folder,
    name="png_converter"
)

# Example: Convert a 3D medical image to PNG slices
# Assuming you have a 3D numpy array or Image object
# For a 3D array of shape (slices, height, width)
image_data = np.random.randint(0, 255, (10, 512, 512), dtype=np.uint8)  # Example data
medical_image = Image(image_data)  # Create Image object

# Convert and save the slices
png_op.convert_and_save(medical_image, output_folder)
```

The operator will save individual PNG files for each slice in the specified output folder, named sequentially (0.png, 1.png, etc.).
