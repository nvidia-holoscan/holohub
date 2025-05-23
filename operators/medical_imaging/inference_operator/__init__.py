"""
Inference Operator Package.

This package provides the InferenceOperator as a base class for running inference in medical imaging pipelines.
The operator handles model loading, execution, and result management.

.. autosummary::
    :toctree: _autosummary

    InferenceOperator
"""

from .inference_operator import InferenceOperator

__all__ = ["InferenceOperator"]
