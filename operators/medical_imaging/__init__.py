# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Medical Imaging Operators Package.

This package provides a collection of operators for medical imaging processing,
including DICOM handling, visualization, and AI model inference.

.. autosummary::
    :toctree: _autosummary

    core
    utils
    exceptions
"""

from . import _version, exceptions as exceptions, utils as utils

# Import all operators
from .clara_viz_operator import ClaraVizOperator

# Import core
from .core import (
    AppContext,
    Application,
    DataPath,
    Image,
    IOType,
    Model,
    ModelFactory,
    NamedModel,
    RuntimeEnv,
    TorchScriptModel,
    TritonModel,
    parse_args,
)
from .dicom_data_loader_operator import DICOMDataLoaderOperator
from .dicom_encapsulated_pdf_writer_operator import DICOMEncapsulatedPDFWriterOperator
from .dicom_seg_writer_operator import DICOMSegmentationWriterOperator, SegmentDescription
from .dicom_series_selector_operator import DICOMSeriesSelectorOperator
from .dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from .dicom_text_sr_writer_operator import DICOMTextSRWriterOperator
from .inference_operator import InferenceOperator
from .monai_bundle_inference_operator import MonaiBundleInferenceOperator
from .monai_seg_inference_operator import MonaiSegInferenceOperator
from .nii_data_loader_operator import NiftiDataLoader
from .png_converter_operator import PNGConverterOperator
from .publisher_operator import PublisherOperator
from .stl_conversion_operator import STLConversionOperator

__version__ = _version.get_versions()["version"]

__all__ = [
    "AppContext",
    "Application",
    "ClaraVizOperator",
    "DICOMDataLoaderOperator",
    "DICOMEncapsulatedPDFWriterOperator",
    "DICOMSegmentationWriterOperator",
    "DICOMSeriesSelectorOperator",
    "DICOMSeriesToVolumeOperator",
    "DICOMTextSRWriterOperator",
    "DataPath",
    "IOType",
    "Image",
    "InferenceOperator",
    "Model",
    "ModelFactory",
    "MonaiBundleInferenceOperator",
    "MonaiSegInferenceOperator",
    "NamedModel",
    "NiftiDataLoader",
    "PNGConverterOperator",
    "PublisherOperator",
    "RuntimeEnv",
    "STLConversionOperator",
    "SegmentDescription",
    "TorchScriptModel",
    "TritonModel",
    "exceptions",
    "parse_args",
    "utils",
]
