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

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Union

from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec

from operators.medical_imaging.core.domain.dicom_series import DICOMSeries
from operators.medical_imaging.core.domain.dicom_series_selection import StudySelectedSeries
from operators.medical_imaging.utils.dicom_utils import (
    EquipmentInfo,
    ModelInfo,
    save_dcm_file,
    write_common_modules,
)
from operators.medical_imaging.utils.importutil import optional_import
from operators.medical_imaging.utils.version import get_sdk_semver

dcmread, _ = optional_import("pydicom", name="dcmread")
dcmwrite, _ = optional_import("pydicom.filewriter", name="dcmwrite")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
Sequence, _ = optional_import("pydicom.sequence", name="Sequence")
PdfReader, _ = optional_import("PyPDF2", name="PdfReader")


class DICOMEncapsulatedPDFWriterOperator(Operator):
    """Class to write DICOM Encapsulated PDF Instance with provided PDF bytes in memory.

    Named inputs:
        pdf_bytes: Bytes of the the PDF content.
        study_selected_series_list: Optional, DICOM series for copying metadata from.

    Named output:
        None

    File output:
        Generated DICOM instance file in the provided output folder.
    """

    # File extension for the generated DICOM Part 10 file.
    DCM_EXTENSION = ".dcm"
    # The default output folder for saving the generated DICOM instance file.
    DEFAULT_OUTPUT_FOLDER = Path(os.getcwd()) / "output"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Union[str, Path],
        model_info: ModelInfo,
        equipment_info: Optional[EquipmentInfo] = None,
        copy_tags: bool = True,
        custom_tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Class to write DICOM Encapsulated PDF Instance with PDF bytes in memory or in a file.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            output_folder (str or Path): The folder for saving the generated DICOM instance file.
            copy_tags (bool): True, default, for copying DICOM attributes from a provided DICOMSeries.
                              If True and no DICOMSeries obj provided, runtime exception is thrown.
            model_info (ModelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info (EquipmentInfo, optional): Object encapsulating info for DICOM Equipment Module.
                                                      Defaults to None.
            custom_tags (Dict[str, str], optional): Dictionary for setting custom DICOM tags using Keywords
                                                    and str values only. Defaults to None.

        Raises:
            ValueError: If copy_tags is true and no DICOMSeries object provided, or
                        if PDF bytes cannot be found in memory or loaded from the file.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        # Need to init the output folder until the execution context supports dynamic FS path
        # Not trying to create the folder to avoid exception on init
        self.output_folder = (
            Path(output_folder)
            if output_folder
            else DICOMEncapsulatedPDFWriterOperator.DEFAULT_OUTPUT_FOLDER
        )
        self.copy_tags = copy_tags
        self.model_info = model_info if model_info else ModelInfo()
        self.equipment_info = equipment_info if equipment_info else EquipmentInfo()
        self.custom_tags = custom_tags
        self.input_name_bytes = "pdf_bytes"
        self.input_name_dcm_series = "study_selected_series_list"

        # Set own Modality and Media Storage SOP Class UID, e.g.,
        #   "DOC" for PDF
        #   "SR" for Structured Report
        #   "1.2.840.10008.5.1.4.1.1.88.11" for Basic Text SR Storage
        #   "1.2.840.10008.5.1.4.1.1.104.1" for Encapsulated PDF Storage,
        #   "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD
        #   "1.2.840.10008.5.1.4.1.1.66.4" for Segmentation Storage
        #   '1.2.840.10008.5.1.4.1.1.104.1' for Encapsulated PDF Storage
        self.modality_type = "DOC"
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.104.1"

        # Equipment version may be different from contributing equipment version
        try:
            self.software_version_number = get_sdk_semver()  # SDK Version
        except Exception:
            self.software_version_number = ""
        self.operators_name = f"AI Algorithm {self.model_info.name}"
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s) if applicable.

        This operator does not have an output for the next operator, rather file output only.

        Args:
            spec (OperatorSpec): The Operator specification for inputs and outputs etc.
        """

        spec.input(self.input_name_bytes)
        spec.input(self.input_name_dcm_series).condition(ConditionType.NONE)  # Optional input

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O.

        For now, only a single result content is supported, which could be in bytes or a path
        to the PDF file. The DICOM series used during inference is optional, but is required if the
        `copy_tags` is true indicating the generated DICOM object needs to copy study level metadata.

        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When bytes are not in the input, and the file is not given or found.
            ValueError: Input bytes and PDF file not in the input, or no DICOM series when required.
            IOError: If fails to get the bytes of the PDF
        """

        # Gets the input, prepares the output folder, and then delegates the processing.
        pdf_bytes: bytes = b""
        pdf_bytes = op_input.receive(self.input_name_bytes)
        if not pdf_bytes or not len(pdf_bytes.strip()):
            raise IOError("Input is read but blank.")

        study_selected_series_list = None
        try:
            study_selected_series_list = op_input.receive(self.input_name_dcm_series)
        except Exception:
            pass

        dicom_series = None  # It can be None if not to copy_tags.
        if self.copy_tags:
            # Get the first DICOM Series for retrieving study level tags.
            if not study_selected_series_list or len(study_selected_series_list) < 1:
                raise ValueError("Missing input, list of 'StudySelectedSeries'.")
            for study_selected_series in study_selected_series_list:
                if not isinstance(study_selected_series, StudySelectedSeries):
                    raise ValueError(
                        "Element in input is not expected type, 'StudySelectedSeries'."
                    )
                for selected_series in study_selected_series.selected_series:
                    dicom_series = selected_series.series
                    break

        # The output folder should come from the execution context when it is supported.
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Now ready to starting writing the DICOM instance
        self.write(pdf_bytes, dicom_series, self.output_folder)

    def write(self, content_bytes, dicom_series: Optional[DICOMSeries], output_dir: Path):
        """Writes DICOM object

        Args:
            content_bytes (bytes): Content bytes of PDF
            dicom_series (DicomSeries): DicomSeries object encapsulating the original series.
            output_dir (Path): Folder path for saving the generated file.
        """
        self._logger.debug("Writing DICOM object...\n")

        if not isinstance(content_bytes, bytes):
            raise ValueError("Input must be bytes.")
        elif not len(content_bytes.strip()):
            raise ValueError("Content is empty.")
        elif not self._is_pdf_bytes(content_bytes):
            raise ValueError("Not PDF bytes.")

        if not isinstance(output_dir, Path):
            raise ValueError("output_dir is not a valid Path.")

        output_dir.mkdir(parents=True, exist_ok=True)  # Just in case

        ds = write_common_modules(
            dicom_series,
            self.copy_tags,
            self.modality_type,
            self.sop_class_uid,
            self.model_info,
            self.equipment_info,
        )

        # Encapsulated PDF specific
        # SC Equipment Module
        ds.Modality = self.modality_type
        ds.ConversionType = "SD"  # Describes the kind of image conversion, Scanned Doc

        # Encapsulated Document Module
        ds.VerificationFlag = "UNVERIFIED"  # Not attested by a legally accountable person.

        ds.BurnedInAnnotation = "YES"
        ds.DocumentTitle = "Generated Observations"
        ds.MIMETypeOfEncapsulatedDocument = "application/pdf"
        ds.EncapsulatedDocument = content_bytes

        # ConceptNameCode Sequence
        seq_concept_name_code = Sequence()
        ds_concept_name_code = Dataset()
        ds_concept_name_code.CodeValue = "18748-4"
        ds_concept_name_code.CodingSchemeDesignator = "LN"
        ds_concept_name_code.CodeMeaning = "Diagnostic Imaging Report"
        seq_concept_name_code.append(ds_concept_name_code)
        ds.ConceptNameCodeSequence = seq_concept_name_code

        # For now, only allow str Keywords and str value
        if self.custom_tags:
            for k, v in self.custom_tags.items():
                if isinstance(k, str) and isinstance(v, str):
                    try:
                        ds.update({k: v})
                    except Exception as ex:
                        # Best effort for now.
                        logging.warning(f"Tag {k} was not written, due to {ex}")

        # Instance file name is the same as the new SOP instance UID
        file_path = output_dir.joinpath(
            f"{ds.SOPInstanceUID}{DICOMEncapsulatedPDFWriterOperator.DCM_EXTENSION}"
        )
        save_dcm_file(ds, file_path)
        self._logger.info(f"DICOM SOP instance saved in {file_path}")

    def _is_pdf_bytes(self, content: bytes):
        try:
            bytes_stream = BytesIO(content)
            reader = PdfReader(bytes_stream)
            self._logger.debug(f"The PDF has {reader.pages} page(s).")
        except Exception as ex:
            self._logger.exception(f"Cannot read as PDF: {ex}")
            return False
        return True


# Commenting out the following as pttype complains about the constructor for no reason
# def test(test_copy_tags: bool = True):
#     from operators.medical_imaging.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
#     from operators.medical_imaging.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

#     current_file_dir = Path(__file__).parent.resolve()
#     dcm_folder = current_file_dir.joinpath("../../../inputs/livertumor_ct/dcm/1-CT_series_liver_tumor_from_nii014")
#     pdf_file = current_file_dir.joinpath("../../../inputs/pdf/TestPDF.pdf")
#     out_path = Path("output_pdf_op").absolute()
#     pdf_bytes = b"Not PDF bytes."

#     fragment = Fragment()
#     loader = DICOMDataLoaderOperator(fragment, name="loader_op")
#     series_selector = DICOMSeriesSelectorOperator(fragment, name="selector_op")
#     sr_writer = DICOMEncapsulatedPDFWriterOperator(
#         fragment,
#         output_folder=out_path,
#         copy_tags=test_copy_tags,
#         model_info=None,
#         equipment_info=EquipmentInfo(),
#         custom_tags={"SeriesDescription": "Report from AI algorithm. Not for clinical use."},
#         name="writer_op",
#     )

#     # Testing with the main entry functions
#     dicom_series = None
#     if test_copy_tags:
#         study_list = loader.load_data_to_studies(Path(dcm_folder).absolute())
#         study_selected_series_list = series_selector.filter(None, study_list)
#         # Get the first DICOM Series, as for now, only expecting this.
#         if not study_selected_series_list or len(study_selected_series_list) < 1:
#             raise ValueError("Missing input, list of 'StudySelectedSeries'.")
#         for study_selected_series in study_selected_series_list:
#             if not isinstance(study_selected_series, StudySelectedSeries):
#                 raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
#             for selected_series in study_selected_series.selected_series:
#                 print(type(selected_series))
#                 dicom_series = selected_series.series
#                 print(type(dicom_series))

#     with open(pdf_file, "rb") as f:
#         pdf_bytes = f.read()

#     sr_writer.write(pdf_bytes, dicom_series, out_path)


# if __name__ == "__main__":
#     test(test_copy_tags=True)
#     test(test_copy_tags=False)
