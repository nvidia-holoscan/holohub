#!/usr/bin/env python3

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

from holoscan.conditions import CountCondition
from holoscan.core import Application
from monai_totalseg_operator import MonaiTotalSegOperator
from pydicom.sr.codedict import codes  # Required for setting SegmentDescription attributes.

from operators.medical_imaging import (
    AppContext,
    DICOMDataLoaderOperator,
    DICOMSegmentationWriterOperator,
    DICOMSeriesSelectorOperator,
    DICOMSeriesToVolumeOperator,
    SegmentDescription,
)

# Labels for the channels/segments
CHANNEL_DEF = {
    "0": "background",
    "1": "spleen",
    "2": "kidney_right",
    "3": "kidney_left",
    "4": "gallbladder",
    "5": "liver",
    "6": "stomach",
    "7": "aorta",
    "8": "inferior_vena_cava",
    "9": "portal_vein_and_splenic_vein",
    "10": "pancreas",
    "11": "adrenal_gland_right",
    "12": "adrenal_gland_left",
    "13": "lung_upper_lobe_left",
    "14": "lung_lower_lobe_left",
    "15": "lung_upper_lobe_right",
    "16": "lung_middle_lobe_right",
    "17": "lung_lower_lobe_right",
    "18": "vertebrae_L5",
    "19": "vertebrae_L4",
    "20": "vertebrae_L3",
    "21": "vertebrae_L2",
    "22": "vertebrae_L1",
    "23": "vertebrae_T12",
    "24": "vertebrae_T11",
    "25": "vertebrae_T10",
    "26": "vertebrae_T9",
    "27": "vertebrae_T8",
    "28": "vertebrae_T7",
    "29": "vertebrae_T6",
    "30": "vertebrae_T5",
    "31": "vertebrae_T4",
    "32": "vertebrae_T3",
    "33": "vertebrae_T2",
    "34": "vertebrae_T1",
    "35": "vertebrae_C7",
    "36": "vertebrae_C6",
    "37": "vertebrae_C5",
    "38": "vertebrae_C4",
    "39": "vertebrae_C3",
    "40": "vertebrae_C2",
    "41": "vertebrae_C1",
    "42": "esophagus",
    "43": "trachea",
    "44": "heart_myocardium",
    "45": "heart_atrium_left",
    "46": "heart_ventricle_left",
    "47": "heart_atrium_right",
    "48": "heart_ventricle_right",
    "49": "pulmonary_artery",
    "50": "brain",
    "51": "iliac_artery_left",
    "52": "iliac_artery_right",
    "53": "iliac_vena_left",
    "54": "iliac_vena_right",
    "55": "small_bowel",
    "56": "duodenum",
    "57": "colon",
    "58": "rib_left_1",
    "59": "rib_left_2",
    "60": "rib_left_3",
    "61": "rib_left_4",
    "62": "rib_left_5",
    "63": "rib_left_6",
    "64": "rib_left_7",
    "65": "rib_left_8",
    "66": "rib_left_9",
    "67": "rib_left_10",
    "68": "rib_left_11",
    "69": "rib_left_12",
    "70": "rib_right_1",
    "71": "rib_right_2",
    "72": "rib_right_3",
    "73": "rib_right_4",
    "74": "rib_right_5",
    "75": "rib_right_6",
    "76": "rib_right_7",
    "77": "rib_right_8",
    "78": "rib_right_9",
    "79": "rib_right_10",
    "80": "rib_right_11",
    "81": "rib_right_12",
    "82": "humerus_left",
    "83": "humerus_right",
    "84": "scapula_left",
    "85": "scapula_right",
    "86": "clavicula_left",
    "87": "clavicula_right",
    "88": "femur_left",
    "89": "femur_right",
    "90": "hip_left",
    "91": "hip_right",
    "92": "sacrum",
    "93": "face",
    "94": "gluteus_maximus_left",
    "95": "gluteus_maximus_right",
    "96": "gluteus_medius_left",
    "97": "gluteus_medius_right",
    "98": "gluteus_minimus_left",
    "99": "gluteus_minimus_right",
    "100": "autochthon_left",
    "101": "autochthon_right",
    "102": "iliopsoas_left",
    "103": "iliopsoas_right",
    "104": "urinary_bladder",
}


class AISegApp(Application):
    def __init__(self, *args, **kwargs):
        """Creates an application instance."""

        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        # Use Commandline options over environment variables to init context.
        app_context: AppContext = Application.init_app_context(self.argv)
        self._logger.debug(f"Begin {self.compose.__name__}")
        self.app_input_path = Path(app_context.input_path)
        self.app_output_path = Path(app_context.output_path).resolve()
        self.model_path = Path(app_context.model_path)

        self._logger.info(
            f"App input and output path: {self.app_input_path}, {self.app_output_path}"
        )

        # The following uses an alternative loader to load dcm from disk
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=self.app_input_path, name="study_loader_op"
        )

        series_selector_op = DICOMSeriesSelectorOperator(
            self, rules=Sample_Rules_Text, name="series_selector_op"
        )
        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")

        # Model specific inference operator, supporting MONAI transforms.
        seg_op = MonaiTotalSegOperator(
            self,
            app_context=app_context,
            output_folder=self.app_output_path / "saved_images_folder",
            model_path=self.model_path,
            name="seg_op",
        )

        # https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
        # User can Look up SNOMED CT codes at, e.g.
        # https://bioportal.bioontology.org/ontologies/SNOMEDCT

        _algorithm_name = "3D segmentation from a CT series"
        _algorithm_family = codes.DCM.ArtificialIntelligence
        _algorithm_version = "0.1.0"

        # To simplify, only use Organ as the dummy category and type, though the body part
        # names will be correct.
        segment_descriptions = []
        for i in range(1, 105):  # seg 1 to 104
            segment_descriptions.append(
                SegmentDescription(
                    segment_label=CHANNEL_DEF.get(str(i), "unknown"),
                    segmented_property_category=codes.SCT.Organ,
                    segmented_property_type=codes.SCT.Organ,
                    algorithm_name=_algorithm_name,
                    algorithm_family=_algorithm_family,
                    algorithm_version=_algorithm_version,
                )
            )

        custom_tags = {"SeriesDescription": "AI generated Seg, not for clinical use."}

        dicom_seg_writer = DICOMSegmentationWriterOperator(
            self,
            segment_descriptions=segment_descriptions,
            custom_tags=custom_tags,
            output_folder=self.app_output_path,
            name="dcm_seg_writer_op",
        )

        self.add_flow(
            study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")}
        )
        self.add_flow(
            series_selector_op,
            series_to_vol_op,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(series_to_vol_op, seg_op, {("image", "image")})

        # Note below the dicom_seg_writer requires two inputs, each coming from a source operator.
        #   Seg writing needs all segment descriptions coded, otherwise fails.
        self.add_flow(
            series_selector_op,
            dicom_seg_writer,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(seg_op, dicom_seg_writer, {("seg_image", "seg_image")})

        self._logger.debug(f"End {self.compose.__name__}")


# This is a sample series selection rule in JSON, simply selecting CT series.
# If the study has more than 1 CT series, then all of them will be selected.
# Please see more detail in DICOMSeriesSelectorOperator.
# For list of string values, e.g. "ImageType": ["PRIMARY", "ORIGINAL"], it is a match if all elements
# are all in the multi-value attribute of the DICOM series.

Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "CT Series",
            "conditions": {
                "StudyDescription": "(.*?)",
                "Modality": "(?i)CT",
                "SeriesDescription": "(.*?)",
                "ImageType": ["PRIMARY", "ORIGINAL"]
            }
        }
    ]
}
"""

if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -m <model file>, for model file path
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    # e.g.
    #     python3 app.py -i input -m model/model.ts
    #
    logging.info(f"Begin {__name__}")

    app = AISegApp()
    app.run()

    logging.info(f"End {__name__}")
