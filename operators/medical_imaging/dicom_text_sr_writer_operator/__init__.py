"""
DICOM Text SR Writer Operator Package.

This package provides the DICOMTextSRWriterOperator for writing DICOM Structured Report instances.
The operator enables encoding textual results into DICOM-compliant Structured Report objects.

.. autosummary::
    :toctree: _autosummary

    DICOMTextSRWriterOperator
"""

from .dicom_text_sr_writer_operator import DICOMTextSRWriterOperator

__all__ = ["DICOMTextSRWriterOperator"]
