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


from os import getcwd, makedirs
from os.path import join
from pathlib import Path
from typing import Union

import numpy as np
from holoscan.core import Fragment, Operator, OperatorSpec

from operators.medical_imaging.core import Image
from operators.medical_imaging.dicom_data_loader_operator import DICOMDataLoaderOperator
from operators.medical_imaging.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator
from operators.medical_imaging.utils.importutil import optional_import

PILImage, _ = optional_import("PIL", name="Image")


class PNGConverterOperator(Operator):
    """
    This operator writes out a 3D Volumetric Image to to a file folder in a slice by slice manner.

    Named input:
        image: Image object or numpy ndarray

    Named output:
        None

    File output:
        Generated PNG image file(s) saved in the provided output folder.
    """

    # The default output folder for saving the generated DICOM instance file.
    DEFAULT_OUTPUT_FOLDER = Path(getcwd()) / "output"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        output_folder: Union[str, Path],
        **kwargs,
    ):
        """Class to write out a 3D Volumetric Image to a file folder in a slice by slice manner.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            output_folder (str or Path): The folder for saving the generated DICOM instance file.
        """

        self.output_folder = (
            output_folder if output_folder else PNGConverterOperator.DEFAULT_OUTPUT_FOLDER
        )
        self.input_name_image = "image"
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input(self.input_name_image)

    def compute(self, op_input, op_output, context):
        input_image = op_input.receive(self.input_name_image)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.convert_and_save(input_image, self.output_folder)

    def convert_and_save(self, image, path):
        """
        extracts the slices in originally acquired direction (often axial)
        and saves them in PNG format slice by slice in the specified directory
        """

        if isinstance(image, Image):
            image_data = image.asnumpy()
        elif isinstance(image, np.ndarray):
            image_data = image
        else:
            raise ValueError(f"Input is not Image or ndarray, {type(image)}.")
        image_shape = image_data.shape

        num_images = image_shape[0]

        for i in range(0, num_images):
            input_data = image_data[i, :, :]
            pil_image = PILImage.fromarray(input_data)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.save(join(str(path), join(str(i) + ".png")))


def main():
    from pathlib import Path

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("../../../inputs/spleen_ct/dcm")
    out_path = "output_png"
    makedirs(out_path, exist_ok=True)

    files = []
    fragment = Fragment()
    loader = DICOMDataLoaderOperator(fragment, name="dcm_loader")
    loader._list_files(
        data_path,
        files,
    )
    study_list = loader._load_data(files)
    series = study_list[0].get_all_series()[0]

    print(f"The loaded series object properties:\n{series}")

    op1 = DICOMSeriesToVolumeOperator(fragment, name="series_to_vol")
    op1.prepare_series(series)
    voxels = op1.generate_voxel_data(series)
    metadata = op1.create_metadata(series)
    image = op1.create_volumetric_image(voxels, metadata)

    print(f"The converted Image object metadata:\n{metadata}")

    op2 = PNGConverterOperator(fragment, output_folder=out_path, name="png_converter")
    # Not mocking the operator context, so bypassing compute
    op2.convert_and_save(image, op2.output_folder)

    print(f"The converted PNG files are in: {out_path}")


if __name__ == "__main__":
    main()
