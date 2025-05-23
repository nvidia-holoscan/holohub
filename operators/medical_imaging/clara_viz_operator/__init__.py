"""
Clara Viz Operator Package.

This package provides the ClaraVizOperator for interactive 3D visualization of medical imaging data.
The operator supports GPU-accelerated rendering and interactive manipulation of volume data and segmentation masks.

.. autosummary::
    :toctree: _autosummary

    ClaraVizOperator
"""

from .clara_viz_operator import ClaraVizOperator

__all__ = ["ClaraVizOperator"]
