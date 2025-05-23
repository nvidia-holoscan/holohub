"""
DICOM Segmentation Writer (All Frames) Operator Package.

This package provides the DICOMSegmentationWriterOperator for writing DICOM Segmentation instances with all frames.
The operator enables encoding segmentation data into DICOM-compliant segmentation objects with multi-frame support.

.. autosummary::
    :toctree: _autosummary

    DICOMSegmentationWriterOperator
"""

from .dicom_seg_writer_operator_all_frames import DICOMSegmentationWriterOperator

__all__ = ["DICOMSegmentationWriterOperator"]
