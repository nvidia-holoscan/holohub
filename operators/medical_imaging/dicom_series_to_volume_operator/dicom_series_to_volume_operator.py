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

import copy
import logging
import math
from typing import Dict, List, Union

import numpy as np
from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec

from operators.medical_imaging.core.domain.dicom_series_selection import StudySelectedSeries
from operators.medical_imaging.core.domain.image import Image


class DICOMSeriesToVolumeOperator(Operator):
    """This operator converts an instance of DICOMSeries into an Image object.

    The loaded Image Object can be used for further processing via other operators.
    The data array will be a 3D image NumPy array with index order of `DHW`.
    Channel is limited to 1 as of now, and `C` is absent in the NumPy array.

    Named Input:
        study_selected_series_list: List of StudySelectedSeries.
    Named Output:
        image: Image object.
    """

    def __init__(self, fragment: Fragment, *args, **kwargs):
        """Create an instance for a containing application object.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
        """

        self.input_name_series = "study_selected_series_list"
        self.output_name_image = "image"
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_series)
        spec.output(self.output_name_image).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator and handles I/O."""

        study_selected_series_list = op_input.receive(self.input_name_series)

        # TODO: need to get a solution to correctly annotate and consume multiple image outputs.
        # For now, only supports the one and only one selected series.
        image = self.convert_to_image(study_selected_series_list)
        op_output.emit(image, self.output_name_image)

    def convert_to_image(
        self, study_selected_series_list: List[StudySelectedSeries]
    ) -> Union[Image, None]:
        """Extracts the pixel data from a DICOM Series and other attributes to create an Image object"""
        # For now, only supports the one and only one selected series.
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing expected input 'study_selected_series_list'")

        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            selection_name = selected_series.selection_name
            self.prepare_series(dicom_series)
            metadata = self.create_metadata(dicom_series)

            # Add to the metadata the DICOMStudy properties and selection metadata
            metadata.update(self._get_instance_properties(study_selected_series.study))
            selection_metadata = {"selection_name": selection_name}
            metadata.update(selection_metadata)

            voxel_data = self.generate_voxel_data(dicom_series)
            image = self.create_volumetric_image(voxel_data, metadata)

            # Now it is time to assign the converted image to the SelectedSeries obj
            selected_series.image = image

            # Break out since limited to one series/image for now
            break

        # TODO: This needs to be updated once allowed to output multiple Image objects
        return study_selected_series_list[0].selected_series[0].image

    def generate_voxel_data(self, series):
        """Applies rescale slope and rescale intercept to the pixels.

        Supports monochrome image only for now. Photometric Interpretation attribute,
        tag (0028,0004), is considered. Both MONOCHROME2 (IDENTITY) and MONOCHROME1 (INVERSE)
        result in an output image where The minimum sample value is intended to be displayed as black.

        Args:
            series: DICOM Series for which the pixel data needs to be extracted.

        Returns:
            A 3D numpy tensor representing the volumetric data.
        """
        slices = series.get_sop_instances()
        # The sop_instance get_pixel_array() returns a 2D NumPy array with index order
        # of `HW`. The pixel array of all instances will be stacked along the first axis,
        # so the final 3D NumPy array will have index order of [DHW]. This is consistent
        # with the NumPy array returned from the ITK GetArrayViewFromImage on the image
        # loaded from the same DICOM series.
        vol_data = np.stack([s.get_pixel_array() for s in slices], axis=0)
        vol_data = vol_data.astype(np.int16)

        # For now we support monochrome image only, for which DICOM Photometric Interpretation
        # (0028,0004) has defined terms, MONOCHROME1 and MONOCHROME2, with the former being:
        #   Pixel data represent a single monochrome image plane. The minimum sample value is
        #   intended to be displayed as white after any VOI gray scale transformations have been
        #   performed. See PS3.4. This value may be used only when Samples per Pixel (0028,0002)
        #   has a value of 1. May be used for pixel data in a Native (uncompressed) or Encapsulated
        #   (compressed) format; see Section 8.2 in PS3.5.
        # and for the latter "The minimum sample value is intended to be displayed as black"
        #
        # In this function, pixel data will be interpreted as if MONOCHROME2, hence inverting
        # MONOCHROME1 for the final voxel data.

        photometric_interpretation = (
            slices[0].get_native_sop_instance().get("PhotometricInterpretation", "").strip().upper()
        )
        presentation_lut_shape = (
            slices[0].get_native_sop_instance().get("PresentationLUTShape", "").strip().upper()
        )

        if not photometric_interpretation:
            logging.warning("Cannot get value of attribute Photometric Interpretation.")

        if photometric_interpretation != "MONOCHROME2":
            if photometric_interpretation == "MONOCHROME1" or presentation_lut_shape == "INVERSE":
                logging.debug("Applying INVERSE transformation as required for MONOCHROME1 image.")
                vol_data = np.amax(vol_data) - vol_data
            else:
                raise ValueError(
                    f"Cannot process pixel data with Photometric Interpretation of {photometric_interpretation}."
                )

        # Rescale Intercept and Slope attributes might be missing, but safe to assume defaults.
        try:
            intercept = slices[0][0x0028, 0x1052].value
        except KeyError:
            intercept = 0

        try:
            slope = slices[0][0x0028, 0x1053].value
        except KeyError:
            slope = 1

        if slope != 1:
            vol_data = slope * vol_data.astype(np.float64)
            vol_data = vol_data.astype(np.int16)
        vol_data += np.int16(intercept)
        return np.array(vol_data, dtype=np.int16)

    def create_volumetric_image(self, vox_data, metadata):
        """Creates an instance of 3D image.

        Args:
            vox_data: A numpy array representing the volumetric data.
            metadata: DICOM attributes in a dictionary.

        Returns:
            An instance of Image object.
        """
        image = Image(vox_data, metadata)
        return image

    def prepare_series(self, series):
        """Computes the slice normal for each slice and then projects the first voxel of each
        slice on that slice normal.

        It computes the distance of that point from the origin of the patient coordinate system along the slice normal.
        It orders the slices in the series according to that distance.

        Args:
            series: An instance of DICOMSeries.
        """

        if len(series._sop_instances) <= 1:
            series.depth_pixel_spacing = (
                1.0  # Default to 1, e.g. for CR image, similar to (Simple) ITK
            )
            return

        slice_indices_to_be_removed = []
        depth_pixel_spacing = 0.0
        last_slice_normal = [0.0, 0.0, 0.0]

        for slice_index, slice in enumerate(series._sop_instances):
            distance = 0.0
            point = [0.0, 0.0, 0.0]
            slice_normal = [0.0, 0.0, 0.0]
            slice_position = None
            cosines = None

            try:
                image_orientation_patient_de = slice[0x0020, 0x0037]
                if image_orientation_patient_de is not None:
                    image_orientation_patient = image_orientation_patient_de.value
                    cosines = image_orientation_patient
            except KeyError:
                pass

            try:
                image_poisition_patient_de = slice[0x0020, 0x0032]
                if image_poisition_patient_de is not None:
                    image_poisition_patient = image_poisition_patient_de.value
                    slice_position = image_poisition_patient
            except KeyError:
                pass

            distance = 0.0

            if (cosines is not None) and (slice_position is not None):
                slice_normal[0] = cosines[1] * cosines[5] - cosines[2] * cosines[4]
                slice_normal[1] = cosines[2] * cosines[3] - cosines[0] * cosines[5]
                slice_normal[2] = cosines[0] * cosines[4] - cosines[1] * cosines[3]

                last_slice_normal = copy.deepcopy(slice_normal)

                i = 0
                while i < 3:
                    point[i] = slice_normal[i] * slice_position[i]
                    i += 1

                distance += point[0] + point[1] + point[2]

                series._sop_instances[slice_index].distance = distance
                series._sop_instances[slice_index].first_pixel_on_slice_normal = point
            else:
                print("going to removing slice ", slice_index)
                slice_indices_to_be_removed.append(slice_index)

        for sl_index, _ in enumerate(slice_indices_to_be_removed):
            del series._sop_instances[sl_index]

        series._sop_instances = sorted(series._sop_instances, key=lambda s: s.distance)
        series.depth_direction_cosine = copy.deepcopy(last_slice_normal)

        if len(series._sop_instances) > 1:
            p1 = series._sop_instances[0].first_pixel_on_slice_normal
            p2 = series._sop_instances[1].first_pixel_on_slice_normal
            depth_pixel_spacing = (
                (p1[0] - p2[0]) * (p1[0] - p2[0])
                + (p1[1] - p2[1]) * (p1[1] - p2[1])
                + (p1[2] - p2[2]) * (p1[2] - p2[2])
            )
            depth_pixel_spacing = math.sqrt(depth_pixel_spacing)
            series.depth_pixel_spacing = depth_pixel_spacing

        s_1 = series._sop_instances[0]
        s_n = series._sop_instances[-1]
        num_slices = len(series._sop_instances)
        self.compute_affine_transform(s_1, s_n, num_slices, series)

    def compute_affine_transform(self, s_1, s_n, n, series):
        """Computes the affine transform for this series. It does it in both DICOM Patient oriented
        coordinate system as well as the pne preferred by NIFTI standard. Accordingly, the two attributes
        dicom_affine_transform and nifti_affine_transform are stored in the series instance.

        The Image Orientation Patient contains two triplets, [rx ry rz cx cy cz], which encode
        direction cosines of the row and column of an image slice. The Image Position Patient of the first slice in
        a volume, [x1 y1 z1], is the x, y, z coordinates of the upper-left corner voxel of the slice. These two
        parameters define the location of the slice in PCS. To determine the location of a volume, the Image
        Position Patient of another slice is normally needed. In practice, we tend to use the position of the last
        slice in a volume, [xn yn zn]. The voxel size within the slice plane, [vr vc], is stored in object Pixel Spacing.

        Args:
            s_1: A first slice in the series.
            s_n: A last slice in the series.
            n: A number of slices in the series.
            series: An instance of DICOMSeries.
        """

        m1 = np.arange(1, 17, dtype=float).reshape(4, 4)
        m2 = np.arange(1, 17, dtype=float).reshape(4, 4)

        image_orientation_patient = None
        try:
            image_orientation_patient_de = s_1[0x0020, 0x0037]
            if image_orientation_patient_de is not None:
                image_orientation_patient = image_orientation_patient_de.value
        except KeyError:
            pass
        rx = image_orientation_patient[0]
        ry = image_orientation_patient[1]
        rz = image_orientation_patient[2]
        cx = image_orientation_patient[3]
        cy = image_orientation_patient[4]
        cz = image_orientation_patient[5]

        vr = 0.0
        vc = 0.0
        try:
            pixel_spacing_de = s_1[0x0028, 0x0030]
            if pixel_spacing_de is not None:
                vr = pixel_spacing_de.value[0]
                vc = pixel_spacing_de.value[1]
        except KeyError:
            pass

        x1 = 0.0
        y1 = 0.0
        z1 = 0.0

        xn = 0.0
        yn = 0.0
        zn = 0.0

        ip1 = None

        try:
            ip1_de = s_1[0x0020, 0x0032]
            ipn_de = s_n[0x0020, 0x0032]
            ip1 = ip1_de.value
            ipn = ipn_de.value

        except KeyError:
            pass

        x1 = ip1[0]
        y1 = ip1[1]
        z1 = ip1[2]

        xn = ipn[0]
        yn = ipn[1]
        zn = ipn[2]

        m1[0, 0] = rx * vr
        m1[0, 1] = cx * vc
        m1[0, 2] = (xn - x1) / (n - 1)
        m1[0, 3] = x1

        m1[1, 0] = ry * vr
        m1[1, 1] = cy * vc
        m1[1, 2] = (yn - y1) / (n - 1)
        m1[1, 3] = y1

        m1[2, 0] = rz * vr
        m1[2, 1] = cz * vc
        m1[2, 2] = (zn - z1) / (n - 1)
        m1[2, 3] = z1

        m1[3, 0] = 0
        m1[3, 1] = 0
        m1[3, 2] = 0
        m1[3, 3] = 1

        series.dicom_affine_transform = m1

        m2[0, 0] = -rx * vr
        m2[0, 1] = -cx * vc
        m2[0, 2] = -(xn - x1) / (n - 1)
        m2[0, 3] = -x1

        m2[1, 0] = -ry * vr
        m2[1, 1] = -cy * vc
        m2[1, 2] = -(yn - y1) / (n - 1)
        m2[1, 3] = -y1

        m2[2, 0] = rz * vr
        m2[2, 1] = cz * vc
        m2[2, 2] = (zn - z1) / (n - 1)
        m2[2, 3] = z1

        m2[3, 0] = 0
        m2[3, 1] = 0
        m2[3, 2] = 0
        m2[3, 3] = 1

        series.nifti_affine_transform = m2

    def create_metadata(self, series) -> Dict:
        """Collects all relevant metadata from the DICOM Series and creates a dictionary.

        Args:
            series: An instance of DICOMSeries.

        Returns:
            An instance of a dictionary containing metadata for the volumetric image.
        """

        # Set metadata with series properties that are not None.
        metadata = {}
        if series:
            metadata = self._get_instance_properties(series)
        return metadata

    @staticmethod
    def _get_instance_properties(obj: object, not_none: bool = True) -> Dict:
        prop_dict = {}
        if obj:
            for attribute in [
                x for x in type(obj).__dict__ if isinstance(type(obj).__dict__[x], property)
            ]:
                attr_val = getattr(obj, attribute, None)
                if not_none:
                    if attr_val is not None:
                        prop_dict[attribute] = attr_val
                else:
                    prop_dict[attribute] = attr_val

        return prop_dict


def test():
    from pathlib import Path

    from operators.medical_imaging.operators.dicom_data_loader_operator import (
        DICOMDataLoaderOperator,
    )
    from operators.medical_imaging.operators.dicom_series_selector_operator import (
        DICOMSeriesSelectorOperator,
    )

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../inputs/spleen_ct/dcm").absolute()

    fragment = Fragment()
    loader = DICOMDataLoaderOperator(fragment, name="loader_op")
    series_selector = DICOMSeriesSelectorOperator(fragment, name="selector_op")
    vol_op = DICOMSeriesToVolumeOperator(fragment, name="series_to_vol_op")

    study_list = loader.load_data_to_studies(data_path)
    study_selected_series_list = series_selector.filter(None, study_list)
    image = vol_op.convert_to_image(study_selected_series_list)

    print(f"Image NumPy array shape (index order DHW): {image.asnumpy().shape}")
    for k, v in image.metadata().items():
        print(f"{(k)}: {(v)}")


if __name__ == "__main__":
    test()
