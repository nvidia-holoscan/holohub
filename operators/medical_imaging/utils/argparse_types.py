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
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def valid_dir_path(path: str) -> Path:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert string input to pathlib.Path if the given path exists and it is a directory path.
    If directory does not exist, create the directory and convert string input to pathlib.Path.

    Args:
        path: string input path

    Returns:
        If path exists and is a directory, return absolute path as a pathlib.Path object.

        If path exists and is not a directory, raises argparse.ArgumentTypeError.

        If path doesn't exist, create the directory and return absolute path as a pathlib.Path object.
    """
    dir_path = Path(path).absolute()
    if dir_path.exists():
        if dir_path.is_dir():
            return dir_path
        else:
            raise argparse.ArgumentTypeError(
                f"Expected directory path: {dir_path!r} is not a directory"
            )

    # create directory
    dir_path.mkdir(parents=True)
    return dir_path


def valid_existing_dir_path(path: str) -> Path:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert string input to pathlib.Path if the given path exists and it is a directory path.

    Args:
        path: string input path

    Returns:
        If path exists and is a directory, return absolute path as a pathlib.Path object.

        If path doesn't exist or it is not a directory, raises argparse.ArgumentTypeError.
    """
    dir_path = Path(path).absolute()
    if dir_path.exists() and dir_path.is_dir():
        return dir_path
    raise argparse.ArgumentTypeError(f"No such directory: {dir_path!r}")


def valid_existing_path(path: str) -> Path:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert string input to pathlib.Path if the given file/folder path exists.

    Args:
        path: string input path

    Returns:
        If path exists, return absolute path as a pathlib.Path object.

        If path doesn't exist, raises argparse.ArgumentTypeError.
    """
    file_path = Path(path).absolute()
    if file_path.exists():
        return file_path
    raise argparse.ArgumentTypeError(f"No such file/folder: {file_path!r}")
