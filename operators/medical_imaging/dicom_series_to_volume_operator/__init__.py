"""
DICOM Series to Volume Operator Package.

This package provides the DICOMSeriesToVolumeOperator for converting DICOM series into volumetric images.
The operator enables construction of volume images suitable for 3D processing and visualization.

.. autosummary::
    :toctree: _autosummary

    DICOMSeriesToVolumeOperator
"""

from .dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

__all__ = ["DICOMSeriesToVolumeOperator"]
