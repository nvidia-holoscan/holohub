# Plan: Update README.md

**Created:** 2025-11-27 10:00 UTC
**Last Modified:** 2025-11-27 10:00 UTC
**Author:** gemini-3-pro-preview

## Goal
Update `README.md` to accurately reflect the current codebase capabilities, structure, and usage, moving away from MVP-specific language while maintaining the "research code" aesthetic.

## Proposed Changes

1.  **Title & Mission**:
    *   Keep the title "Ultrasound Post-Processing".
    *   Refine the mission statement.
    *   Remove "WIP" from the title if appropriate, or keep it subtle.

2.  **Quick Start**:
    *   Simplify installation instructions (using `uv`).
    *   Clear commands for running the Streamlit GUI.
    *   Clear commands for running the Holoscan app (with optional synthetic data).

3.  **Project Structure**:
    *   Add a section describing the key directories:
        *   `ultra_post/core`: Pipeline logic and data loading.
        *   `ultra_post/ops`: Image processing operators.
        *   `ultra_post/app`: Streamlit and Holoscan applications.
        *   `presets`: YAML configurations for pipelines.

4.  **Core Concepts**:
    *   Briefly explain the **Pipeline** architecture (DAG of operators).
    *   Explain **Presets** (YAML-defined processing chains).

5.  **CLI Utilities**:
    *   Document `uspp list-ops`.
    *   Document `uspp validate-preset`.

6.  **Authoring Guide**:
    *   Streamline the "Add a new operator" section.
    *   Streamline the "Create a preset" section.

7.  **Integration**:
    *   Mention the `i4h-sensor-simulation` integration for synthetic data generation.

## Verification
*   Verify all commands listed in the README actually work (conceptually, based on `pyproject.toml`).
*   Ensure directory paths are correct.

