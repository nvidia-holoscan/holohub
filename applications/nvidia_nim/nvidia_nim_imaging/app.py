# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import hashlib
import logging
import os
import pathlib

import requests
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

from operators.unzip.unzip_op import UnzipOp

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


class PrintMessageOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("data")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("data")

        if data:
            print(f"File saved in: {data}")


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
        base_url=None,
        api_key=None,
        **kwargs,
    ):
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
        op_output.emit(response.content, "file_name")


class NimImaging(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        nim_op = NIMOperator(
            self,
            CountCondition(self, count=1),
            name="nim",
            **self.kwargs("nim"),
        )

        unzip_op = UnzipOp(
            self, CountCondition(self, count=1), name="unzip", filter="*.nrrd", output_path="."
        )
        print_files_op = PrintMessageOp(self, CountCondition(self, count=1), name="print_files")

        self.add_flow(nim_op, unzip_op, {("file_name", "zip_file_bytes")})
        self.add_flow(unzip_op, print_files_op, {("matching_files", "data")})

    def set_nifti_file_path(self, file):
        self._nifti_file_path = file
        logger.info(f"NIFTI file: {self._nifti_file_path}")


def _download_dataset(api_key, validate_file_checksum):
    logger.info("Downloading sample image data from NVIDIA NGC...")
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.get(payload["image"], headers=headers)
    response.raise_for_status()
    nifti_filename = os.path.abspath("sample.nii.gz")

    with open(nifti_filename, "wb") as f:
        f.write(response.content)

    if (
        validate_file_checksum
        and hashlib.md5(open(nifti_filename, "rb").read()).hexdigest()
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
        "-v",
        "--validate-file-checksum",
        action="store",
        default=True,
        type=bool,
        dest="validate_file_checksum",
        help="Validate the checksum of the downloaded file",
    )

    args = parser.parse_args()
    app = NimImaging()
    app.config(str(args.config))

    api_key = get_api_key(app)
    nifti_file = _download_dataset(api_key, args.validate_file_checksum)
    app.set_nifti_file_path(nifti_file)

    try:
        app.run()
    except Exception as e:
        logger.error("Error:", str(e))
