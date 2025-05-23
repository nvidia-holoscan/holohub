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

fragment = Fragment()
pub_op = PublisherOperator(fragment, ...)
```

## Acknowledgements

Developed by NVIDIA Holoscan SDK Team. See LICENSE for details.
