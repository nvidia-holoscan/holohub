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
from pathlib import Path
from typing import List

from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec

from operators.medical_imaging.core.domain.dicom_series import DICOMSeries
from operators.medical_imaging.core.domain.dicom_study import DICOMStudy
from operators.medical_imaging.exceptions import ItemNotExistsError
from operators.medical_imaging.utils.importutil import optional_import

dcmread, _ = optional_import("pydicom", name="dcmread")
get_testdata_file, _ = optional_import("pydicom.data", name="dcmread")
FileSet, _ = optional_import("pydicom.fileset", name="FileSet")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
valuerep, _ = optional_import("pydicom", name="valuerep")
InvalidDicomError, _ = optional_import("pydicom.errors", name="InvalidDicomError")


class DICOMDataLoaderOperator(Operator):
    """This operator loads DICOM studies into memory from a folder of DICOM instance files.

    Named Input:
        input_folder: Path to the folder containing DICOM instance files. Optional and not requiring input.
                      If present, data from this input will be used as the input folder of DICOM instance files.

    Name Output:
        dicom_study_list: A list of DICOMStudy objects in memory. The name can be changed via attribute, `output_name`.
    """

    DEFAULT_INPUT_FOLDER = Path.cwd() / "input"
    DEFAULT_OUTPUT_NAME = "dicom_study_list"
    SOP_CLASSES_TO_IGNORE = [
        "1.2.840.10008.1.3.10",  # Media Storage Directory Storage, aka DICOMDIR
    ]

    # For now, need to have the input folder as an instance attribute, set on init, because even there is the optional
    # named input to receive data containing the path, there might not be upstream operator to emit the data.
    def __init__(
        self,
        fragment: Fragment,
        *args,
        input_folder: Path = DEFAULT_INPUT_FOLDER,
        output_name: str = DEFAULT_OUTPUT_NAME,
        must_load: bool = True,
        **kwargs,
    ):
        """Creates an instance of this class.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            input_folder (Path): Folder containing DICOM instance files to load from.
                                 Defaults to `input` in the current working directory.
                                 Can be overridden by via the named input receiving from other's output.
            output_name (str): The name for the output, which is list of DICOMStudy objects.
                               Defaults to `dicom_study_list`, and if None or blank passed in.
            must_load (bool): If true, raise exception if no study is loaded.
                              Defaults to True.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self._must_load = must_load
        self.input_path = input_folder
        self.index = 0
        self.input_name = "input_folder"
        self.output_name = (
            output_name.strip()
            if output_name and len(output_name.strip()) > 0
            else DICOMDataLoaderOperator.DEFAULT_OUTPUT_NAME
        )

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name).condition(
            ConditionType.NONE
        )  # Optional input, not requiring upstream emitter.
        spec.output(self.output_name)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handlesI/O."""

        self.index += 1
        input_path = None
        try:
            input_path = op_input.receive(self.input_name)
        except Exception:
            pass

        if not input_path or not Path(input_path).is_dir():
            self._logger.info(
                f"No or invalid input path from the optional input port: {input_path}"
            )
            # Use the object attribute if it is valid
            if self.input_path and self.input_path.is_dir():
                input_path = self.input_path
            else:
                raise ValueError(
                    f"No valid input path from input port or obj attribute: {self.input_path}"
                )

        dicom_study_list = self.load_data_to_studies(input_path)
        op_output.emit(dicom_study_list, self.output_name)

    def load_data_to_studies(self, input_path: Path):
        """Load DICOM data from files into DICOMStudy objects in a list.

        It scans through the input directory for all SOP instances.
        It groups them by a collection of studies where each study contains one or more series.
        This method returns a list of studies.
        If there is no studies loaded, an exception will be thrown if set to must load.

        Args:
            input_path (Path): The folder containing DICOM instance files.

        Returns:
            List[DICOMStudy]: List of DICOMStudy.

        Raises:
            ValueError: If the folder to load files from does not exist.
            ItemNotExistsError: If no studies loaded and must_load is True.
        """
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError("Required input folder does not exist.")

        files: List[str] = []
        self._list_files(input_path, files)
        dicom_studies = self._load_data(files)
        if self._must_load and len(dicom_studies) < 1:
            raise ItemNotExistsError(f"No study loaded from path {input_path}.")

        return dicom_studies

    def _list_files(self, path, files: List[str]):
        """Collects fully qualified names of all files recursively given a directory path.

        Args:
            path: A directory containing DICOM SOP instances. It may have nested hirerarchical directories.
            files: This method populates "files" with fully qualified names of files that belong to the specified directory.
        """
        for item in os.listdir(path):
            item = os.path.join(path, item)
            if os.path.isdir(item):
                self._list_files(item, files)
            else:
                files.append(item)

    def _load_data(self, files: List[str]):
        """Provides a list of DICOM Studies given a list of fully qualified file names.

        Args:
            files: A list of file names that represents SOP Instances

        Returns:
            A list of DICOMStudy objects.
        """
        study_dict = {}
        series_dict = {}
        sop_instances = []

        for file in files:
            try:
                sop_instances.append(dcmread(file))
            except InvalidDicomError as ex:
                self._logger.warn(f"Ignored {file}, reason being: {ex}")

        for sop_instance in sop_instances:
            study_instance_uid = sop_instance[0x0020, 0x000D].value.name  # name is the UID as str

            # First need to eliminate the SOP instances whose SOP Class is to be ignored.
            if "SOPInstanceUID" not in sop_instance:
                self._logger.warn("Instance ignored due to missing SOP instance UID tag")
                continue
            sop_instance_uid = sop_instance["SOPInstanceUID"].value
            if "SOPClassUID" not in sop_instance:
                self._logger.warn(
                    f"Instance ignored due to missing SOP Class UID tag, {sop_instance_uid}"
                )
                continue
            if sop_instance["SOPClassUID"].value in DICOMDataLoaderOperator.SOP_CLASSES_TO_IGNORE:
                self._logger.warn(
                    f"Instance ignored for being in the ignored class, {sop_instance_uid}"
                )
                continue

            if study_instance_uid not in study_dict:
                study = DICOMStudy(study_instance_uid)
                self.populate_study_attributes(study, sop_instance)
                study_dict[study_instance_uid] = study

            series_instance_uid = sop_instance[0x0020, 0x000E].value.name  # name is the UID as str

            if series_instance_uid not in series_dict:
                series = DICOMSeries(series_instance_uid)
                series_dict[series_instance_uid] = series
                self.populate_series_attributes(series, sop_instance)
                study_dict[study_instance_uid].add_series(series)

            series_dict[series_instance_uid].add_sop_instance(sop_instance)
        return list(study_dict.values())

    def populate_study_attributes(self, study, sop_instance):
        """Populates study level attributes in the study data structure.

        Args:
            study: A DICOM Study instance that needs to be filled-in with study level attribute values
            sop_instance: A sample DICOM SOP Instance that contains the list of attributed which will be parsed
        """
        try:
            study_id_de = sop_instance[0x0020, 0x0010]
            if study_id_de is not None:
                study.StudyID = study_id_de.value
        except KeyError:
            pass

        try:
            study_date_de = sop_instance[0x0008, 0x0020]
            if study_date_de is not None:
                study.StudyDate = study_date_de.value
        except KeyError:
            pass

        try:
            study_time_de = sop_instance[0x0008, 0x0030]
            if study_time_de is not None:
                study.StudyTime = study_time_de.value
        except KeyError:
            pass

        try:
            study_desc_de = sop_instance[0x0008, 0x1030]
            if study_desc_de is not None:
                study.StudyDescription = study_desc_de.value
        except KeyError:
            pass

        try:
            accession_number_de = sop_instance[0x0008, 0x0050]
            if accession_number_de is not None:
                study.AccessionNumber = accession_number_de.value
        except KeyError:
            pass

    def populate_series_attributes(self, series, sop_instance):
        """Populates series level attributes in the study data structure.

        Args:
            study: A DICOM Series instance that needs to be filled-in with series level attribute values
            sop_instance: A sample DICOM SOP Instance that contains the list of attributed which will be parsed
        """
        try:
            series_date_de = sop_instance[0x0008, 0x0021]
            if series_date_de is not None:
                series.SeriesDate = series_date_de.value
        except KeyError:
            pass

        try:
            series_time_de = sop_instance[0x0008, 0x0031]
            if series_time_de is not None:
                series.SeriesTime = series_time_de.value
        except KeyError:
            pass

        try:
            series_modality_de = sop_instance[0x0008, 0x0060]
            if series_modality_de is not None:
                series.Modality = series_modality_de.value
        except KeyError:
            pass

        try:
            series_description_de = sop_instance[0x0008, 0x103E]
            if series_description_de is not None:
                series.SeriesDescription = series_description_de.value
        except KeyError:
            pass

        try:
            body_part_examined_de = sop_instance[0x0008, 0x0015]
            if body_part_examined_de is not None:
                series.BodyPartExamined = body_part_examined_de.value
        except KeyError:
            pass

        try:
            patient_position_de = sop_instance[0x0018, 0x5100]
            if patient_position_de is not None:
                series.PatientPosition = patient_position_de.value
        except KeyError:
            pass

        try:
            series_number_de = sop_instance[0x0020, 0x0011]
            if series_number_de is not None:
                val = series_number_de.value
                if isinstance(val, valuerep.IS):
                    series.SeriesNumber = val.real
                else:
                    series.SeriesNumber = int(val)
        except KeyError:
            pass

        try:
            laterality_de = sop_instance[0x0020, 0x0060]
            if laterality_de is not None:
                series.Laterality = laterality_de.value
        except KeyError:
            pass

        try:
            tag_pixel_spacing = "PixelSpacing"  # tag (0x0028, 0x0030)
            tag_imager_pixel_spacing = "ImagerPixelSpacing"  # tag (0x0018, 0x1164)
            if tag_pixel_spacing in sop_instance:
                pixel_spacing_de = sop_instance[tag_pixel_spacing]
            elif tag_imager_pixel_spacing in sop_instance:
                pixel_spacing_de = sop_instance[tag_imager_pixel_spacing]
            else:
                raise KeyError(
                    "Neither {tag_pixel_spacing} nor {tag_imager_pixel_spacing} in dataset header."
                )
            if pixel_spacing_de is not None:
                series.row_pixel_spacing = pixel_spacing_de.value[0]
                series.col_pixel_spacing = pixel_spacing_de.value[1]
        except KeyError:
            pass

        try:
            image_orientation_paient_de = sop_instance[0x0020, 0x0037]
            if image_orientation_paient_de is not None:
                orientation_orig = image_orientation_paient_de.value
                orientation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for index, _ in enumerate(orientation_orig):
                    orientation[index] = float(orientation_orig[index])

                series.row_direction_cosine = orientation[0:3]
                series.col_direction_cosine = orientation[3:6]

        except KeyError:
            pass


def test():
    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../inputs/spleen_ct/dcm")

    loader = DICOMDataLoaderOperator(Fragment())
    study_list = loader.load_data_to_studies(data_path.absolute())

    for study in study_list:
        print("###############################")
        print(study)
        for series in study.get_all_series():
            print(series)
        for series in study.get_all_series():
            for sop in series.get_sop_instances():
                print("Demonstrating ways to access DICOM attributes in a SOP instance.")
                # No need to get the native_ds = sop.get_native_sop_instance()
                # sop = sop.get_native_sop_instance()
                print(f"   'StudyInstanceUID': {sop['StudyInstanceUID'].repval}")
                print(f"     (0x0020, 0x000D): {sop[0x0020, 0x000D].repval}")
                print(f"   'SeriesInstanceUID': {sop['SeriesInstanceUID'].value.name}")
                print(f"     (0x0020, 0x000E): {sop[0x0020, 0x000E].value.name}")
                print(f"     'SOPInstanceUID': {sop['SOPInstanceUID'].value.name}")
                print(f"          (0008,0018): {sop[0x0008, 0x0018].value.name}")
                try:
                    print(f"     'InstanceNumber': {sop['InstanceNumber'].repval}")
                    print(f"         (0020, 0013): {sop[0x0020, 0x0013].repval}")
                except KeyError:
                    pass
                # Need to get pydicom dataset to use properties and get method of a dict.
                ds = sop.get_native_sop_instance()
                print(
                    f"   'StudyInstanceUID': {ds.StudyInstanceUID if ds.StudyInstanceUID else ''}"
                )
                print(
                    f"   'SeriesDescription': {ds.SeriesDescription if ds.SeriesDescription else ''}"
                )
                print(
                    "   'IssuerOfPatientID':"
                    f" {ds.get('IssuerOfPatientID', '').repval if ds.get('IssuerOfPatientID', '') else '' }"
                )
                try:
                    print(
                        f"   'IssuerOfPatientID': {ds.IssuerOfPatientID if ds.IssuerOfPatientID else '' }"
                    )
                except AttributeError:
                    print(
                        "    If the IssuerOfPatientID does not exist, ds.IssuerOfPatientID would throw AttributeError."
                    )
                    print("    Use ds.get('IssuerOfPatientID', '') instead.")

                break
            break
    # Test raising exception, or not, depending on if set to must_load.
    non_dcm_dir = current_file_dir.parent / "utils"
    print(f"Test loading from dir without dcm files: {non_dcm_dir}")
    try:
        loader.load_data_to_studies(non_dcm_dir)
    except ItemNotExistsError as ex:
        print(f"Test passed: exception when no studies loaded & must_load flag is True: {ex}")

    relaxed_loader = DICOMDataLoaderOperator(Fragment(), must_load=False)
    study_list = relaxed_loader.load_data_to_studies(non_dcm_dir)
    print(f"Test passed: {len(study_list)} study loaded and is OK when must_load flag is False.")


if __name__ == "__main__":
    test()
