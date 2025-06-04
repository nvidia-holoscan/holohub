"""
NIfTI Data Loader Operator Package.

This package provides the NiftiDataLoader for loading NIfTI medical images.
The operator enables reading NIfTI files and making them available as in-memory images.

.. autosummary::
    :toctree: _autosummary

    NiftiDataLoader
"""

from .nii_data_loader_operator import NiftiDataLoader

__all__ = ["NiftiDataLoader"]
