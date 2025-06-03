"""
DICOM Segmentation Writer Operator

The operator enables encoding segmentation data into DICOM-compliant segmentation objects.

.. autosummary::
    :toctree: _autosummary

    DICOMSegmentationWriterOperator
"""

from .dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription

__all__ = ["DICOMSegmentationWriterOperator", "SegmentDescription"]
