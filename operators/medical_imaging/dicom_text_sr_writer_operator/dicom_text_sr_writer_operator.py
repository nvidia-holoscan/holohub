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


class DICOMTextSRWriterOperator(Operator):
    """Class to write DICOM Text SR Instance with provided text input.

    Named inputs:
        text: text content to be encapsulated in a DICOM instance file.
        study_selected_series_list: Optional, DICOM series for copying metadata from.

    Named output:
        None

    File output:
        Generated DICOM instance file in the provided output folder.
    """

    # File extension for the generated DICOM Part 10 file.
    DCM_EXTENSION = ".dcm"
    # The default output folder for saving the generated DICOM instance file.
    # DEFAULT_OUTPUT_FOLDER = Path(os.path.join(os.path.dirname(__file__))) / "output"
    DEFAULT_OUTPUT_FOLDER = Path.cwd() / "output"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Union[str, Path],
        model_info: ModelInfo,
        copy_tags: bool = True,
        equipment_info: Optional[EquipmentInfo] = None,
        custom_tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Class to write DICOM SR SOP Instance for AI textual result in memory or in a file.

        Args:
            output_folder (str or Path): The folder for saving the generated DICOM instance file.
            copy_tags (bool): True, default, for copying DICOM attributes from a provided DICOMSeries.
                              If True and no DICOMSeries obj provided, runtime exception is thrown.
            model_info (ModelInfo): Object encapsulating model creator, name, version and UID.
            equipment_info (EquipmentInfo, optional): Object encapsulating info for DICOM Equipment Module.
                                                      Defaults to None.
            custom_tags (Dict[str, str], optional): Dictionary for setting custom DICOM tags using Keywords and str values only.
                                                    Defaults to None.

        Raises:
            ValueError: If copy_tags is true and no DICOMSeries object provided, or
                        if result cannot be found either in memory or from file.
        """
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        # Need to init the output folder until the execution context supports dynamic FS path
        # Not trying to create the folder to avoid exception on init
        self.output_folder = (
            Path(output_folder)
            if output_folder
            else DICOMTextSRWriterOperator.DEFAULT_OUTPUT_FOLDER
        )
        self.copy_tags = copy_tags
        self.model_info = model_info if model_info else ModelInfo()
        self.equipment_info = equipment_info if equipment_info else EquipmentInfo()
        self.custom_tags = custom_tags
        self.input_name_text = "text"
        self.input_name_dcm_series = "study_selected_series_list"

        # Set own Modality and SOP Class UID e.g.,
        #   "SR" for Structured Report
        #   "1.2.840.10008.5.1.4.1.1.88.11" for Basic Text SR Storage
        #   "1.2.840.10008.5.1.4.1.1.104.1" for Encapsulated PDF Storage,
        #   "1.2.840.10008.5.1.4.1.1.88.34" for Comprehensive 3D SR IOD
        #   "1.2.840.10008.5.1.4.1.1.66.4" for Segmentation Storage
        self.modality_type = "SR"
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
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

        spec.input(self.input_name_text)
        spec.input(self.input_name_dcm_series).condition(ConditionType.NONE)  # Optional input

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O.

        For now, only a single result content is supported, which could be in memory or an accessible file.
        The DICOM series used during inference is optional, but is required if the
        `copy_tags` is true indicating the generated DICOM object needs to copy study level metadata.

        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When result object not in the input, and result file not found either.
            ValueError: Content object and file path not in the inputs, or no DICOM series when required.
            IOError: If the input content is blank.
        """

        # Gets the input, prepares the output folder, and then delegates the processing.
        result_text = str(op_input.receive(self.input_name_text)).strip()
        if not result_text:
            raise IOError("Input is read but blank.")

        study_selected_series_list = None
        try:
            study_selected_series_list = op_input.receive(self.input_name_dcm_series)
        except Exception:
            pass

        dicom_series = None  # It can be None if not to copy_tags.
        if self.copy_tags:
            # Get the first DICOM Series to retrieve study level tags.
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
        self.write(result_text, dicom_series, self.output_folder)

    def write(self, content_text, dicom_series: Optional[DICOMSeries], output_dir: Path):
        """Writes DICOM object

        Args:
            content_file (str): file containing the contents
            dicom_series (DicomSeries): DicomSeries object encapsulating the original series.
            model_info (MoelInfo): Object encapsulating model creator, name, version and UID.

        Returns:
            PyDicom Dataset
        """
        self._logger.debug("Writing DICOM object...\n")

        if not content_text or not len(content_text.strip()):
            raise ValueError("Content is empty.")
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

        # SR specific
        ds.VerificationFlag = "UNVERIFIED"  # Not attested by a legally accountable person.

        # Per recommendation of IHE Radiology Technical Framework Supplement
        # AI Results (AIR) Rev1.1 - Trial Implementation
        # Specifically for Qualitative Findings,
        # Qualitative findings shall be encoded in an instance of the DICOM Comprehensive 3D SR SOP
        # Class using TID 1500 (Measurement Report) as the root template.
        # DICOM PS3.16: TID 1500 Measurement Report
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_A.html#sect_TID_1500
        # The value for Procedure Reported (121058, DCM, "Procedure reported") shall describe the
        # imaging procedure analyzed, not the algorithm used.

        # Use text value for example
        ds.ValueType = "CONTAINER"

        # ConceptNameCode Sequence
        seq_concept_name_code = Sequence()
        ds.ConceptNameCodeSequence = seq_concept_name_code

        # Concept Name Code Sequence: Concept Name Code
        ds_concept_name_code = Dataset()
        ds_concept_name_code.CodeValue = "18748-4"
        ds_concept_name_code.CodingSchemeDesignator = "LN"
        ds_concept_name_code.CodeMeaning = "Diagnostic Imaging Report"
        seq_concept_name_code.append(ds_concept_name_code)

        ds.ContinuityOfContent = "SEPARATE"

        # Content Sequence
        content_sequence = Sequence()
        ds.ContentSequence = content_sequence

        # Content Sequence: Content 1
        content1 = Dataset()
        content1.RelationshipType = "CONTAINS"
        content1.ValueType = "TEXT"

        # Concept Name Code Sequence
        concept_name_code_sequence = Sequence()
        content1.ConceptNameCodeSequence = concept_name_code_sequence

        # Concept Name Code Sequence: Concept Name Code 1
        concept_name_code1 = Dataset()
        concept_name_code1.CodeValue = "111412"  # or 111413 "Overall Assessment"
        concept_name_code1.CodingSchemeDesignator = "DCM"
        concept_name_code1.CodeMeaning = "Narrative Summary"  # or 111413 'Overall Assessment'
        concept_name_code_sequence.append(concept_name_code1)

        content1.TextValue = content_text  # The actual report content text
        content_sequence.append(content1)

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
            f"{ds.SOPInstanceUID}{DICOMTextSRWriterOperator.DCM_EXTENSION}"
        )
        save_dcm_file(ds, file_path)
        self._logger.info(f"DICOM SOP instance saved in {file_path}")


# Commenting out the following as pttype complains about the constructor for no reason
# def test(test_copy_tags: bool = True):
#     from operators.medical_imaging.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
#     from operators.medical_imaging.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator

#     current_file_dir = Path(__file__).parent.resolve()
#     data_path = current_file_dir.joinpath("../../../inputs/livertumor_ct/dcm/1-CT_series_liver_tumor_from_nii014")
#     out_path = Path("output_sr_op").absolute()
#     test_report_text = "Tumors detected in Liver using MONAI Liver Tumor Seg model."

#     fragment = Fragment()
#     loader = DICOMDataLoaderOperator(fragment, name="loader_op")
#     series_selector = DICOMSeriesSelectorOperator(fragment, name="selector_op")
#     sr_writer = DICOMTextSRWriterOperator(
#         fragment,
#         output_folder=out_path,
#         copy_tags=test_copy_tags,
#         model_info=None,
#         equipment_info=EquipmentInfo(),
#         custom_tags={"SeriesDescription": "Textual report from AI algorithm. Not for clinical use."},
#         name="sr_writer"
#     )

#     # Testing with the main entry functions
#     dicom_series = None
#     if test_copy_tags:
#         study_list = loader.load_data_to_studies(Path(data_path).absolute())
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

#     sr_writer.write(test_report_text, dicom_series, out_path)


# if __name__ == "__main__":
#     test(True)
#     test(False)
