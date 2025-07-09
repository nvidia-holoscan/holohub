# MONAI Segmentation Inference Operator

This segmentation operator uses MONAI transforms and Sliding Window Inference to segment medical images.

## Overview

The `MonaiSegInferenceOperator` performs pre-transforms on input images, runs segmentation inference using a specified model, and applies post-transforms. The segmentation result is returned as an in-memory image object and can optionally be saved to disk.

## Requirements

- Holoscan SDK Python package
- MONAI
- torch

## Example Usage

```python
from pathlib import Path
import torch
from monai.transforms import Compose, LoadImage, ScaleIntensity, EnsureChannelFirst
from holoscan.core import Fragment
from operators.medical_imaging.monai_segmentation_inference_operator import MonaiSegInferenceOperator
from operators.medical_imaging.core import AppContext, IOMapping, IOType, Image

# Initialize the fragment
fragment = Fragment()

# Create app context
app_context = AppContext({})

# Define transforms
pre_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
])

post_transforms = Compose([
    # Add your post-processing transforms here
])

# Initialize the segmentation operator
seg_op = MonaiSegInferenceOperator(
    fragment,
    roi_size=(96, 96, 96),  # Example ROI size for 3D images
    pre_transforms=pre_transforms,
    post_transforms=post_transforms,
    app_context=app_context,
    model_name="unet",  # Example model name
    overlap=0.25,
    sw_batch_size=4,
    model_path=Path("/path/to/your/model.pt")  # Replace with your model path
)
```
