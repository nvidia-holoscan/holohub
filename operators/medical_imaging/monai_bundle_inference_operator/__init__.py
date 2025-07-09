"""
MONAI Bundle Inference Operator Package.

This package provides the MonaiBundleInferenceOperator for running inference using MONAI Bundles.
The operator enables loading and executing MONAI Bundle models for medical imaging tasks.

.. autosummary::
    :toctree: _autosummary

    MonaiBundleInferenceOperator
"""

from .monai_bundle_inference_operator import MonaiBundleInferenceOperator

__all__ = ["MonaiBundleInferenceOperator"]
