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

import argparse
import gzip
import hashlib
import io
import json
import logging
import os
import pathlib
import shutil
import tempfile
import zipfile
from typing import Dict

import requests
from halo import Halo
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, HolovizOp
from holoscan.resources import UnboundedAllocator

from holohub.volume_loader import VolumeLoaderOp
from holohub.volume_renderer import VolumeRendererOp

logger = logging.getLogger("httpx")
logger.setLevel(logging.WARN)
logger = logging.getLogger("openai")
logger.setLevel(logging.WARN)
logger = logging.getLogger("NVIDIA_NIM_CHAT")
logging.basicConfig(level=logging.INFO)

sample = "example-1"
payload = {
    "image": f"https://assets.ngc.nvidia.com/products/api-catalog/vista3d/{sample}.nii.gz",
    "prompts": {"classes": ["liver", "spleen"]},
}


class MessageBody:
    def __init__(self, model_params: Dict, user_input: str):
        self.model_params = model_params
        self.user_input = user_input

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_text):
        return MessageBody(**json.loads(json_text))


def get_api_key(app):
    api_key = app.kwargs("nim")["api_key"]
    if not api_key:
        api_key = os.getenv("API_KEY", None)

    if not api_key:
        logger.warning("Setting up connection without an API key.")
        logger.warning(
            "Set 'api-key' in the nvidia_nim.yaml config file or set the environment variable 'API_KEY'."
        )
        print("")
    return api_key


class NIMOperator(Operator):
    def __init__(
        self,
        fragment,
        *args,
        name,
        spinner,
        base_url=None,
        api_key=None,
        **kwargs,
    ):
        self.spinner = spinner
        self.base_url = base_url
        self.api_key = api_key
        self.model_params = dict(kwargs)

        if self.api_key is None:
            self.api_key = get_api_key(fragment)

        # Need to call the base class constructor last
        super().__init__(fragment, name=name)

    def setup(self, spec: OperatorSpec):
        spec.output("file_name")

    def compute(self, op_input, op_output, context):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        session = requests.Session()
        response = session.post(self.base_url, headers=headers, json=payload)

        response.raise_for_status()
        seg_filename = os.path.abspath(f"{sample}_seg.nrrd")
        with tempfile.TemporaryDirectory() as temp_dir:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(temp_dir)

            shutil.move(os.path.join(temp_dir, os.listdir(temp_dir)[0]), seg_filename)

        assert os.path.isfile(seg_filename) and os.access(
            seg_filename, os.R_OK
        ), f"File {seg_filename} doesn't exist or isn't readable"

        logger.info(f"Segfile saved to {seg_filename}")
        op_output.emit(seg_filename, "file_name", "std::string")


class NimImaging(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        spinner = Halo(text="thinking...", spinner="dots")

        nim_op = NIMOperator(
            self,
            CountCondition(self, count=1),
            name="nim",
            spinner=spinner,
            **self.kwargs("nim"),
        )

        volume_allocator = UnboundedAllocator(self, name="allocator")
        logger.info(f"NIFTI file: {self._nifti_file_path}")
        nifti_volume_loader = VolumeLoaderOp(
            self,
            name="nifti_volume_loader",
            file_name=self._nifti_file_path,
            allocator=volume_allocator,
        )

        mask_volume_loader = VolumeLoaderOp(
            self,
            CountCondition(self, count=1),
            name="mask_volume_loader",
            allocator=volume_allocator,
        )

        volume_renderer = VolumeRendererOp(
            self,
            name="volume_renderer",
            config_file=self._rendering_config,
            allocator=volume_allocator,
            alloc_width=512,
            alloc_height=512,
        )

        # Python is not supporting gxf::VideoBuffer, need to convert the video buffer received
        # from the volume renderer to a tensor.
        volume_renderer_format_converter = FormatConverterOp(
            self,
            name="volume_renderer_format_converter",
            pool=volume_allocator,
            in_dtype="rgba8888",
            out_dtype="rgba8888",
        )
        visualizer = HolovizOp(self, name="viz", **self.kwargs("holoviz"))

        self.add_flow(nim_op, mask_volume_loader, {("file_name", "file_name")})
        self.add_flow(
            nifti_volume_loader,
            volume_renderer,
            {
                ("volume", "density_volume"),
                ("spacing", "density_spacing"),
                ("permute_axis", "density_permute_axis"),
                ("flip_axes", "density_flip_axes"),
            },
        )
        self.add_flow(
            mask_volume_loader,
            volume_renderer,
            {
                ("volume", "mask_volume"),
                ("spacing", "mask_spacing"),
                ("permute_axis", "mask_permute_axis"),
                ("flip_axes", "mask_flip_axes"),
            },
        )

        self.add_flow(
            volume_renderer,
            volume_renderer_format_converter,
            {("color_buffer_out", "source_video")},
        )

        self.add_flow(volume_renderer_format_converter, visualizer, {("tensor", "receivers")})

    def set_nifti_file_path(self, file):
        self._nifti_file_path = file

    def set_rendering_config(self, file):
        self._rendering_config = file


def _download_dataset(api_key):
    logger.info("Downloading sample image data from NVIDIA NGC...")
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.get(payload["image"], headers=headers)
    response.raise_for_status()
    nifti_filename = os.path.abspath("sample.nii.gz")
    buffer = io.BytesIO(response.content)
    with open(nifti_filename, "wb") as f:
        f.write(response.content)
    if (
        hashlib.md5(open(nifti_filename, "rb").read()).hexdigest()
        != "56bed2308a195b4cdbb3a875bcf113a2"
    ):
        raise ValueError("File checksum did not match.")

    logger.info(f"NIFTI file saved to {nifti_filename}")
    return nifti_filename

def valid_existing_path(path: str) -> pathlib.Path:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert string input to pathlib.Path if the given file/folder path exists.

    Args:
        path: string input path

    Returns:
        If path exists, return absolute path as a pathlib.Path object.

        If path doesn't exist, raises argparse.ArgumentTypeError.
    """
    path = os.path.expanduser(path)
    file_path = pathlib.Path(path).absolute()
    if file_path.exists():
        return file_path
    raise argparse.ArgumentTypeError(f"No such file/folder: '{file_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVIDIA Inference Microservice (NIM) - Vista-3D ")
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default=os.path.join(os.path.dirname(__file__), "nvidia_nim.yaml"),
        type=valid_existing_path,
        dest="config",
        help="Application configuration file",
    )
    parser.add_argument(
        "-r",
        "--render-config",
        action="store",
        default="../../../../../data/nvidia_nim_imaging/config.json",
        type=valid_existing_path,
        dest="render_config",
        help="Transfer function config file",
    )

    args = parser.parse_args()
    app = NimImaging()
    app.config(str(args.config))

    api_key = get_api_key(app)
    nifti_file = _download_dataset(api_key)
    app.set_nifti_file_path(nifti_file)
    app.set_rendering_config(str(args.render_config))

    try:
        app.run()
    except Exception as e:
        logger.error("Error:", str(e))
