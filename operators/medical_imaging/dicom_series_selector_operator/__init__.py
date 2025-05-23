"""
DICOM Series Selector Operator Package.

This package provides the DICOMSeriesSelectorOperator for selecting specific DICOM series from a set of studies.
The operator enables filtering and selection of relevant DICOM series for downstream processing.

.. autosummary::
    :toctree: _autosummary

    DICOMSeriesSelectorOperator
"""

from .dicom_series_selector_operator import DICOMSeriesSelectorOperator

__all__ = ["DICOMSeriesSelectorOperator"]
