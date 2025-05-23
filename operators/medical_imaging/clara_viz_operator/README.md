# Clara Viz Operator

This operator integrates Clara Viz visualization into medical imaging pipelines.

## Overview

The `ClaraVizOperator` enables advanced visualization of medical imaging data using Clara Viz, supporting GPU-accelerated rendering and interaction.

## Requirements

- Holoscan SDK Python package
- clara-viz

## Example Usage

```python
from holoscan.core import Fragment
from operators.medical_imaging.clara_viz_operator import ClaraVizOperator

fragment = Fragment()
viz_op = ClaraVizOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
