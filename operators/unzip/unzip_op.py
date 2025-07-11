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

import glob
import io
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from holoscan.core import Fragment, Operator, OperatorSpec

logger = logging.getLogger("unzip")


class UnzipOp(Operator):
    """
    Unzips a zip file, extracts files with matching criteria to a directory and emits the file paths.
    """

    def __init__(
        self,
        fragment: Fragment,
        *args,
        filter="",
        output_path=None,
        **kwargs,
    ):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.filter = filter
        self.output_path = Path(output_path).resolve()
        # Call the base class __init__() last.
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("zip_file_bytes")
        spec.output("matching_files")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("zip_file_bytes")
        matching_files = self._unzip(data)
        if len(matching_files) == 0:
            logger.warn("No matching files found")
        else:
            files = ":".join(matching_files)
            op_output.emit(files, "matching_files", "std::string")
            logger.info(f"Found matching file(s) {files}")

    def _unzip(self, data):
        matching_files = []
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Extracting zip file to {temp_dir}")
            z = zipfile.ZipFile(io.BytesIO(data))
            z.extractall(temp_dir)
            glob_path = f"{temp_dir}/{self.filter}"
            print(f"Searching files in {glob_path}...")
            for file in glob.glob(glob_path):
                file_path = Path(file).resolve()
                dest = os.path.join(self.output_path, file_path.name)
                shutil.move(file, dest)
                matching_files.append(dest)
                logger.debug(f"File added {dest}")

        return matching_files
