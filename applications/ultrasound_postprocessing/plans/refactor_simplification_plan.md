Ultrasound Post-Processing — "Academic Code Golf" Refactor Plan
===============================================================

Created: 2025-11-27
Last Modified: 2025-11-27
By: gemini-3-pro-preview

Goals
-----
- **Simplicity Wins**: Reduce codebase size and complexity to the absolute minimum ("Academic Code Golf").
- **Independence**: Mathematical operations (`ops/`) must be pure functions with zero framework dependencies.
- **Readability**: Explicit is better than implicit. No auto-discovery magic or complex class hierarchies.
- **Flexibility**: Pipelines are just data (lists of dictionaries), runnable by any engine (Streamlit, Holoscan, CLI).

Key Principles
--------------
- **Pure Functions**: Operators are simple Python functions (`def nlm(img, h=0.1)`) using CuPy. No classes, no decorators.
- **Explicit Registry**: Operators are registered in `ultra_post/ops/registry.py`. No runtime directory scanning.
- **Generic Pipelines**: A pipeline is a list of `(op_name, params)`. It is not tied to Holoscan or Streamlit.
- **Dumb UI**: The UI reflects the function signatures (via `DEFAULT_PARAMS`), not the other way around.

Architecture
------------
1.  **`ops/` (The Math)**:
    -   Contains only pure signal processing logic.
    -   Dependencies: `cupy`, `cupyx`, `scipy` (optional).
    -   Example: `nlm.py`, `filters.py`.

2.  **`core/pipeline.py` (The Config)**:
    -   **REFACTOR**: Delete the `Pipeline` class and `run()` method.
    -   Replace with simple YAML load/save functions that return `list[dict]`.

3.  **`app/holoscan_app.py` (The Graph)**:
    -   **NEW**: Iterate through the config list.
    -   Dynamically wrap each function into a `HoloscanOp`.
    -   Chain them together: `Source -> Op1 -> Op2 -> ... -> Viz`.

4.  **`ops/registry.py` (The Truth)**:
    -   Exports `OPS` (name -> function mapping).
    -   Exports `DEFAULT_PARAMS` (name -> dict mapping for UI defaults).

Implementation Status
---------------------
- [x] **Purify Ops**: Ops in `ultra_post/ops/` are pure functions.
- [x] **Kill Complexity**: `ultra_post/core/base.py` has been deleted.
- [x] **Explicit Registry**: `ultra_post/ops/registry.py` exists.
- [x] **Simplify Pipeline**: `ultra_post/core/pipeline.py` class has been removed, replaced by type alias and functions.
- [x] **Update Apps**: `holoscan_app.py` now dynamically builds the graph.
- [x] **YAML Loading**: `pipeline.py` now handles loading/dumping simple dict lists.

Future Work
-----------
- **Documentation**: Generate simple Markdown docs from function docstrings.
- **New Ops**: Add more advanced filters (e.g., Wavelet denoising).
- **Tests**: Add unit tests for the pure ops and pipeline loading.
