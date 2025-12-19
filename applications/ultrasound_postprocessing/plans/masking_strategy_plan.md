# Masking Strategy Implementation Plan

**Created:** 2025-11-27 19:57 PST
**Last Modified:** 2025-11-27 20:15 PST
**By:** gemini-3-pro-preview

## Overview

Scan-converted ultrasound images have a non-rectangular support (the "Matte") with a black background. We need to prevent this background from bleeding into the tissue during filtering, without modifying every single filter operator.

## Strategy: The `AutoMatte` Wrapper

We will implement a **Wrapper Operator** called `AutoMatte`. It acts as a container for a sub-pipeline of filters.

**Structure:**
```yaml
- op: AutoMatte
  params:
    ops:  # Sub-pipeline
      - op: GaussianFilter
        params: { sigma: 2 }
      - op: MedianFilter
        params: { size: 3 }
```

**Logic Flow (Skip Connection):**
1.  **Pre-Process (Top Cap):**
    *   Detect the Matte (background) using **Flood Fill** from the corners (avoids filling internal anechoic cysts).
    *   Compute **Fill Indices** (nearest valid pixel) using Distance Transform (`edt`).
    *   **Cache** these (Stateful) so they are only computed once per geometry.
    *   **Fill** the background of the input image.
2.  **Process (Middle):**
    *   Run the standard `run_pipeline` on the defined `ops` using the filled image.
    *   The filters operate on "infinite tissue" texture, avoiding edge artifacts.
3.  **Post-Process (Bottom Cap):**
    *   Re-apply the cached Matte to the result, restoring the clean black background.

## Implementation Plan

### 1. Utility Functions (`ultra_post/ops/utils.py`)

*   `get_matte_mask(image)`:
    *   Flood-fill from corners to find connected background.
    *   Returns boolean mask (True = Valid Tissue).
*   `get_fill_indices(mask)`:
    *   `distance_transform_edt` to find nearest valid pixels.
*   `apply_fill(image, indices)`:
    *   Replaces invalid pixels with their nearest neighbors.

### 2. The Operator (`ultra_post/ops/matte.py`)

Create `class AutoMatte`:
*   **Init:** Accepts `ops` (list of dicts). calls `pipeline_from_dict` to create the sub-pipeline.
*   **Call:**
    *   Manages the caching of `_mask` and `_fill_indices`.
    *   Executes the Fill -> Run Pipeline -> Mask sequence.

### 3. Registry (`ultra_post/ops/registry.py`)

*   Register `AutoMatte`.

## Advantages
*   **Encapsulation:** All masking logic is in one file.
*   **Clean Filters:** No changes needed to `filters.py` or `bilateral.py`.
*   **Efficiency:** Expensive mask calculation happens once; per-frame overhead is just memory copy/indexing.
*   **Correctness:** Flood fill ensures cysts aren't filled.
