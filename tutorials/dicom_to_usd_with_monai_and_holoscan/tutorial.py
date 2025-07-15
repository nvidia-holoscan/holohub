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

# Python built-in
import argparse
import logging
import os
import sys
from pathlib import Path

from monai.deploy.conditions import CountCondition
from monai.deploy.core import AppContext, Application

# MONAI
# Required for setting SegmentDescription attributes. Direct import as this is not part of App SDK package.
from monai.deploy.core.domain import Image
from monai.deploy.core.io_type import IOType
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from monai.deploy.operators.monai_bundle_inference_operator import (
    BundleConfigNames,
    IOMapping,
    MonaiBundleInferenceOperator,
)
from monai.deploy.operators.stl_conversion_operator import STLConversionOperator

# OpenUSD
from pxr import Kind, Usd, UsdGeom

# HoloHub internal imports
from operators.mesh_to_usd.SendMeshToUSDOp import SendMeshToUSDOp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DICOMToUSDTutorial")


class AISpleenSegmentationApp(Application):
    """Run AI segmentation with MONAI Deploy and write to an OpenUSD file"""

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
                    "Modality": "(?i)CT",
                    "ImageType": ["PRIMARY", "ORIGINAL"],
                    "PhotometricInterpretation": "MONOCHROME2"
                }
            }
        ]
    }
    """

    def __init__(self, *args, existing_stage, **kwargs):
        """Creates an application instance."""
        self._existing_stage = existing_stage
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        self._logger.info(f"Begin {self.compose.__name__}")

        # Use Commandline options over environment variables to init context.
        app_context: AppContext = Application.init_app_context("")
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path)

        # Create the custom operator(s) as well as SDK built-in operator(s).
        study_loader_op = DICOMDataLoaderOperator(
            self, CountCondition(self, 1), input_folder=app_input_path, name="study_loader_op"
        )
        series_selector_op = DICOMSeriesSelectorOperator(
            self, rules=self.Sample_Rules_Text, name="series_selector_op"
        )
        series_to_vol_op = DICOMSeriesToVolumeOperator(self, name="series_to_vol_op")

        # Create the inference operator that supports MONAI Bundle and automates the inference.
        # The IOMapping labels match the input and prediction keys in the pre and post processing.
        # The model_name is optional when the app has only one model.
        # The bundle_path argument optionally can be set to an accessible bundle file path in the dev
        # environment, so when the app is packaged into a MAP, the operator can complete the bundle parsing
        # during init.

        config_names = BundleConfigNames(config_names=["inference"])  # Same as the default

        bundle_spleen_seg_op = MonaiBundleInferenceOperator(
            self,
            input_mapping=[IOMapping("image", Image, IOType.IN_MEMORY)],
            output_mapping=[IOMapping("pred", Image, IOType.IN_MEMORY)],
            app_context=app_context,
            bundle_config_names=config_names,
            name="bundle_spleen_seg_op",
        )

        # Create the processing pipeline, by specifying the source and destination operators, and
        # ensuring the output from the former matches the input of the latter, in both name and type.
        self.add_flow(
            study_loader_op, series_selector_op, {("dicom_study_list", "dicom_study_list")}
        )
        self.add_flow(
            series_selector_op,
            series_to_vol_op,
            {("study_selected_series_list", "study_selected_series_list")},
        )
        self.add_flow(series_to_vol_op, bundle_spleen_seg_op, {("image", "image")})

        stl_binary_file_path = app_output_path.joinpath("stl/mesh.stl")
        stl_conversion_op = STLConversionOperator(
            self, output_file=stl_binary_file_path, name="stl_conversion_op"
        )
        self.add_flow(bundle_spleen_seg_op, stl_conversion_op, {("pred", "image")})

        nt_op = SendMeshToUSDOp(
            self, CountCondition(self, 1), g_stage=self._existing_stage, name="nt"
        )
        self.add_flow(stl_conversion_op, nt_op, {("stl_bytes", "stl_bytes")})
        logging.info(f"End {self.compose.__name__}")


if __name__ == "__main__":
    # Creates the app and test it standalone.
    parser = argparse.ArgumentParser(
        description="DICOM-to-USD MONAI Deploy and Holoscan sample tutorial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        help="OpenUSD output filepath in tutorial container: path/to/spleen-segmentation.usd",
        required=False,
    )
    args = parser.parse_args()
    destination_path = (args.output or f"{os.getcwd()}/output/spleen-segmentation.usd").strip()

    if not destination_path.endswith(".usd"):
        logger.error(f"Expected output filepath ending in `.usd` but received {destination_path}")
        sys.exit(1)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    logger.info(f"Creating stage: {destination_path}")

    existing_stage = Usd.Stage.CreateNew(destination_path)
    UsdGeom.SetStageUpAxis(existing_stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(existing_stage, 0.01)
    # Set the /World prim as the default prim
    default_prim_name = "/World"
    UsdGeom.Xform.Define(existing_stage, default_prim_name)
    default_prim = existing_stage.GetPrimAtPath(default_prim_name)
    existing_stage.SetDefaultPrim(default_prim)
    # Set the default prim as an assembly to support using component references
    Usd.ModelAPI(default_prim).SetKind(Kind.Tokens.assembly)

    logging.debug(f"Stage: {existing_stage}")

    myapp = AISpleenSegmentationApp(existing_stage=existing_stage)
    myapp.run()

    logger.info(f"USD file is available at {destination_path}")
