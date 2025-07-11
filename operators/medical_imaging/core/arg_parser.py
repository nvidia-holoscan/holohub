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
import json
import logging.config
from pathlib import Path
from typing import List, Optional, Union

from operators.medical_imaging.utils import argparse_types

LOG_CONFIG_FILENAME = "logging.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses the arguments passed to the application.

    Args:
        argv (Optional[List[str]], optional): The command line arguments to parse.
            The first item should be the path to the python executable.
            If not specified, ``sys.argv`` is used. Defaults to None.

    Returns:
        argparse.Namespace: parsed arguments.
    """
    if argv is None:
        import sys

        argv = sys.argv
    argv = list(argv)  # copy argv for manipulation to avoid side-effects

    # We have intentionally not set the default using `default="INFO"` here so that the default
    # value from here doesn't override the value in `LOG_CONFIG_FILENAME` unless the user indends to do
    # so. If the user doesn't use this flag to set log level, this argument is set to "None"
    # and the logging level specified in `LOG_CONFIG_FILENAME` is used.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=argparse_types.valid_existing_path,
        help="Path to input folder/file (default: input)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse_types.valid_dir_path,
        help="Path to output folder (default: output)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=argparse_types.valid_existing_path,
        help="Path to model(s) folder/file (default: models)",
    )
    parser.add_argument(
        "--workdir",
        "-w",
        type=argparse_types.valid_dir_path,
        help="Path to workspace folder (default: A temporary '.monai_workdir' folder in the current folder)",
    )

    args = parser.parse_args(argv[1:])
    args.argv = argv  # save argv for later use in runpy

    return args


def set_up_logging(level: Optional[str], config_path: Union[str, Path] = LOG_CONFIG_FILENAME):
    """Initializes the logger and sets up logging level.

    Args:
        level (str): A logging level (DEBUG, INFO, WARN, ERROR, CRITICAL).
        log_config_path (str): A path to logging config file.
    """
    # Default log config path
    log_config_path = Path(__file__).absolute().parent.parent / LOG_CONFIG_FILENAME

    config_path = Path(config_path)

    # If a logging config file that is specified by `log_config_path` exists in the current folder,
    # it overrides the default one
    if config_path.exists():
        log_config_path = config_path

    config_dict = json.loads(log_config_path.read_bytes())

    if level is not None and "root" in config_dict:
        config_dict["root"]["level"] = level
    logging.config.dictConfig(config_dict)
