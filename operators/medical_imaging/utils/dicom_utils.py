# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
import logging
from pathlib import Path
from random import randint
from typing import Any, Optional

from operators.medical_imaging.core.domain.dicom_series import DICOMSeries
from operators.medical_imaging.utils.importutil import optional_import
from operators.medical_imaging.utils.version import get_sdk_semver

dcmread, _ = optional_import("pydicom", name="dcmread")
dcmwrite, _ = optional_import("pydicom.filewriter", name="dcmwrite")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset_, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")


# To address mypy complaint
Dataset: Any = Dataset_

__all__ = [
    "EquipmentInfo",
    "ModelInfo",
    "random_with_n_digits",
    "save_dcm_file",
    "write_common_modules",
]


class ModelInfo(object):
    """Class encapsulating AI model information, according to IHE AI Results (AIR) Rev 1.1.

    The attributes of the class will be used to populate the Contributing Equipment Sequence in the DICOM IOD
    per IHE AIR Rev 1.1, Section 6.5.3.1 General Result Encoding Requirements, as the following,

    The Creator shall describe each algorithm that was used to generate the results in the
    Contributing Equipment Sequence (0018,A001). Multiple items may be included. The Creator
    shall encode the following details in the Contributing Equipment Sequence,
        - Purpose of Reference Code Sequence (0040,A170) shall be (Newcode1, 99IHE, 1630 "Processing Algorithm")
        - Manufacturer (0008,0070)
        - Manufacturer's Model Name (0008,1090)
        - Software Versions (0018,1020)
        - Device UID (0018,1002)

    Each time an AI Model is modified, for example by training, it would be appropriate to update
    the Device UID.
    """

    def __init__(self, creator: str = "", name: str = "", version: str = "", uid: str = ""):
        self.creator = creator if isinstance(creator, str) else ""
        self.name = name if isinstance(name, str) else ""
        self.version = version if isinstance(version, str) else ""
        self.uid = uid if isinstance(uid, str) else ""


class EquipmentInfo(object):
    """Class encapsulating attributes required for DICOM Equipment Module."""

    def __init__(
        self,
        manufacturer: str = "MONAI Deploy",
        manufacturer_model: str = "MONAI Deploy App SDK",
        series_number: str = "0000",
        software_version_number: str = "",
    ):
        self.manufacturer = manufacturer if isinstance(manufacturer, str) else ""
        self.manufacturer_model = manufacturer_model if isinstance(manufacturer_model, str) else ""
        self.series_number = series_number if isinstance(series_number, str) else ""
        if software_version_number:
            self.software_version_number = str(software_version_number)[0:15]
        else:
            try:
                version_str = get_sdk_semver()  # SDK Version
            except Exception:
                version_str = ""  # Fall back to the initial version
            self.software_version_number = version_str[0:15]


# Utility functions


def random_with_n_digits(n):
    """Random number generator to generate n digit int, where 1 <= n <= 32."""

    assert isinstance(n, int) and n <= 32, "Argument n must be an int, n <= 32."
    n = n if n >= 1 else 1
    range_start = 10 ** (n - 1)
    range_end = (10**n) - 1
    return randint(range_start, range_end)


def save_dcm_file(data_set: Dataset, file_path: Path, validate_readable: bool = True):
    """Save a DICOM data set, in pydicom Dataset, to the provided file path."""

    logging.debug(f"DICOM dataset to be written:{data_set}")

    if not isinstance(data_set, Dataset):
        raise ValueError("data_set is not the expected Dataset type.")

    if not str(file_path).strip():
        raise ValueError("file_path to save dcm file not provided.")

    dcmwrite(str(file_path).strip(), data_set, write_like_original=False)
    logging.info(f"Finished writing DICOM instance to file {file_path}")

    if validate_readable:
        # Test reading back
        _ = dcmread(str(file_path))


def write_common_modules(
    dicom_series: Optional[DICOMSeries],
    copy_tags: bool,
    modality_type: str,
    sop_class_uid: str,
    model_info: Optional[ModelInfo] = None,
    equipment_info: Optional[EquipmentInfo] = None,
) -> Dataset:
    """Writes DICOM object common modules with or without a reference DCIOM Series

    Common modules include Study, Patient, Equipment, Series, and SOP common.

    Args:
        dicom_series (DicomSeries): DicomSeries object encapsulating the original series.
        copy_tags (bool): If true, dicom_series must be provided for copying tags.
        modality_type (str): DICOM Modality Type, e.g. SR.
        sop_class_uid (str): Media Storage SOP Class UID, e.g. "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD.
        model_info (MoelInfo): Object encapsulating model creator, name, version and UID.
        equipment_info(EquipmentInfo): Object encapsulating attributes for DICOM Equipment Module

    Returns:
        pydicom Dataset

    Raises:
        ValueError: When dicom_series is not a DICOMSeries object, and new_study is not True.
    """

    if copy_tags:
        if not isinstance(dicom_series, DICOMSeries):
            raise ValueError("A DICOMSeries object is required for coping tags.")

        if len(dicom_series.get_sop_instances()) < 1:
            raise ValueError("DICOMSeries must have at least one SOP instance.")

        # Get one of the SOP instance's native sop instance dataset
        orig_ds = dicom_series.get_sop_instances()[0].get_native_sop_instance()

    logging.debug("Writing DICOM common modules...")

    # Get and format date and time per DICOM standards.
    dt_now = datetime.datetime.now()
    date_now_dcm = dt_now.strftime("%Y%m%d")
    time_now_dcm = dt_now.strftime("%H%M%S")
    offset_from_utc = (
        dt_now.astimezone().isoformat()[-6:].replace(":", "")
    )  # '2022-09-27T22:36:20.143857-07:00'

    # Generate UIDs and descriptions
    my_sop_instance_uid = generate_uid()
    my_series_instance_uid = generate_uid()
    my_series_description = "CAUTION: Not for Diagnostic Use, for research use only."
    my_series_number = str(random_with_n_digits(4))  # 4 digit number to avoid conflict
    my_study_instance_uid = orig_ds.StudyInstanceUID if copy_tags else generate_uid()

    # File meta info data set
    file_meta = Dataset()
    file_meta.FileMetaInformationGroupLength = 198
    file_meta.FileMetaInformationVersion = bytes("01", "utf-8")  # '\x00\x01'

    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = my_sop_instance_uid
    file_meta.TransferSyntaxUID = (
        ImplicitVRLittleEndian  # 1.2.840.10008.1.2, Little Endian Implicit VR
    )
    file_meta.ImplementationClassUID = "1.2.40.0.13.1.1.1"  # Made up. Not registered.
    file_meta.ImplementationVersionName = (
        equipment_info.software_version_number if equipment_info else ""
    )

    # Write modules to data set
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_implicit_VR = True
    ds.is_little_endian = True

    # Content Date (0008,0023) and Content Time (0008,0033) are defined to be the date and time that
    # the document content creation started. In the context of analysis results, these may be considered
    # to be the date and time that the analysis that generated the result(s) started executing.
    # Use current time for now, but could potentially use the actual inference start time.
    ds.ContentDate = date_now_dcm
    ds.ContentTime = time_now_dcm
    ds.TimezoneOffsetFromUTC = offset_from_utc

    # The date and time that the original generation of the data in the document started.
    ds.AcquisitionDateTime = date_now_dcm + time_now_dcm  # Result has just been created.

    # Patient Module, mandatory.
    # Copy over from the original DICOM metadata.
    ds.PatientName = orig_ds.get("PatientName", "") if copy_tags else ""
    ds.PatientID = orig_ds.get("PatientID", "") if copy_tags else ""
    ds.IssuerOfPatientID = orig_ds.get("IssuerOfPatientID", "") if copy_tags else ""
    ds.PatientBirthDate = orig_ds.get("PatientBirthDate", "") if copy_tags else ""
    ds.PatientSex = orig_ds.get("PatientSex", "") if copy_tags else ""

    # Study Module, mandatory
    # Copy over from the original DICOM metadata.
    ds.StudyDate = orig_ds.get("StudyDate", "") if copy_tags else date_now_dcm
    ds.StudyTime = orig_ds.get("StudyTime", "") if copy_tags else time_now_dcm
    ds.AccessionNumber = orig_ds.get("AccessionNumber", "") if copy_tags else ""
    ds.StudyDescription = orig_ds.get("StudyDescription", "") if copy_tags else "AI results."
    ds.StudyInstanceUID = my_study_instance_uid
    ds.StudyID = orig_ds.get("StudyID", "") if copy_tags else "1"
    ds.ReferringPhysicianName = orig_ds.get("ReferringPhysicianName", "") if copy_tags else ""

    # Equipment Module, mandatory
    if equipment_info:
        ds.Manufacturer = equipment_info.manufacturer
        ds.ManufacturerModelName = equipment_info.manufacturer_model
        ds.SeriesNumber = equipment_info.series_number
        ds.SoftwareVersions = equipment_info.software_version_number

    # SOP Common Module, mandatory
    ds.InstanceCreationDate = date_now_dcm
    ds.InstanceCreationTime = time_now_dcm
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = my_sop_instance_uid
    ds.InstanceNumber = "1"
    ds.SpecificCharacterSet = "ISO_IR 100"

    # Series Module, mandatory
    ds.Modality = modality_type
    ds.SeriesInstanceUID = my_series_instance_uid
    ds.SeriesNumber = my_series_number
    ds.SeriesDescription = my_series_description
    ds.SeriesDate = date_now_dcm
    ds.SeriesTime = time_now_dcm
    # Body part copied over, although not mandatory depending on modality
    ds.BodyPartExamined = orig_ds.get("BodyPartExamined", "") if copy_tags else ""
    ds.RequestedProcedureID = orig_ds.get("RequestedProcedureID", "") if copy_tags else ""

    # Contributing Equipment Sequence
    # The Creator shall describe each algorithm that was used to generate the results in the
    # Contributing Equipment Sequence (0018,A001). Multiple items may be included. The Creator
    # shall encode the following details in the Contributing Equipment Sequence:
    #  • Purpose of Reference Code Sequence (0040,A170) shall be (Newcode1, 99IHE, 1630 "Processing Algorithm")
    #  • Manufacturer (0008,0070)
    #  • Manufacturer’s Model Name (0008,1090)
    #  • Software Versions (0018,1020)
    #  • Device UID (0018,1002)

    if model_info:
        # First create the Purpose of Reference Code Sequence
        seq_purpose_of_reference_code = Sequence()
        ds_purpose_of_reference_code = Dataset()
        ds_purpose_of_reference_code.CodeValue = "Newcode1"
        ds_purpose_of_reference_code.CodingSchemeDesignator = "99IHE"
        ds_purpose_of_reference_code.CodeMeaning = '"Processing Algorithm'
        seq_purpose_of_reference_code.append(ds_purpose_of_reference_code)

        seq_contributing_equipment = Sequence()
        ds_contributing_equipment = Dataset()
        ds_contributing_equipment.PurposeOfReferenceCodeSequence = seq_purpose_of_reference_code
        # '(121014, DCM, “Device Observer Manufacturer")'
        ds_contributing_equipment.Manufacturer = model_info.creator
        # u'(121015, DCM, “Device Observer Model Name")'
        ds_contributing_equipment.ManufacturerModelName = model_info.name
        # u'(111003, DCM, “Algorithm Version")'
        ds_contributing_equipment.SoftwareVersions = model_info.version
        ds_contributing_equipment.DeviceUID = (
            model_info.uid
        )  # u'(121012, DCM, “Device Observer UID")'
        seq_contributing_equipment.append(ds_contributing_equipment)
        ds.ContributingEquipmentSequence = seq_contributing_equipment

    logging.debug("DICOM common modules written:\n{}".format(ds))

    return ds
