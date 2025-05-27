# Clara Viz Operator

This operator integrates Clara Viz visualization into medical imaging pipelines.

## Overview

The `ClaraVizOperator` enables advanced visualization of medical imaging data using Clara Viz, supporting GPU-accelerated rendering and interaction.

## Requirements

- Holoscan SDK Python package
- clara-viz
- IPython
- ipywidgets

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.clara_viz_operator import ClaraVizOperator

fragment = Fragment()
viz_op = ClaraVizOperator(
    fragment,
    name="clara_viz",  # Optional operator name
    input_name_image="image",  # Name of the input port for the image
    input_name_seg_image="seg_image"  # Name of the input port for the segmentation image
)
```
