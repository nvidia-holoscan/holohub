"""
MONAI Segmentation Inference Operator Package.

This package provides the MonaiSegInferenceOperator for performing segmentation inference using MONAI.
The operator supports pre-transforms, sliding window inference, and post-transforms for medical image segmentation.

.. autosummary::
    :toctree: _autosummary

    MonaiSegInferenceOperator
"""

from .monai_seg_inference_operator import InfererType, InMemImageReader, MonaiSegInferenceOperator

__all__ = ["MonaiSegInferenceOperator", "InfererType", "InMemImageReader"]
