#!/bin/bash
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

from .factory import ModelFactory
from .model import Model

logger = logging.getLogger(__name__)


class NamedModel(Model):
    """Represents named models in the model repository.

    MONAI Deploy models are stored in a directory structure like this:

        <model-repository-path>/
            <model-name>/
                <model-definition-file>
            <model-name>/
                <model-definition-file>

    This class checks if the given path meets the folder structure of the named model repository:

      1) The path should be a folder path.
      2) The folder should contain only sub folders (model folders).
      3) Each model folder must contain only one model definition file or folder.

    Model items identified would have a file path to the model file.
    """

    model_type: str = "named-model"

    def __init__(self, path: str, name: str = ""):
        """Initializes a named model repository.

        This assumes that the given path is a valid named model repository.

        Args:
            path (str): A Path to the model repository.
            name (str): A name of the model.
        """
        super().__init__(path, name)

        # Clear existing model item and fill model items
        self._items.clear()
        model_path: Path = Path(path)

        for model_folder in model_path.iterdir():
            if model_folder.is_dir():
                # Pick one file (assume that only one file exists in the folder)
                model_file = next(model_folder.iterdir())
                if model_file:
                    # Recursive call to identify the model type
                    model = ModelFactory.create(str(model_file), model_folder.name)
                    # Add model to items only if the model's class is not 'Model'
                    if model and model.__class__ != Model:
                        self._items[model_folder.name] = model

    @classmethod
    def accept(cls, path: str):
        model_path: Path = Path(path)

        # 1) The path should be a folder path.
        if not model_path.is_dir():
            return False, None

        # 2) The folder should contain only sub folders (model folders).
        if not all((p.is_dir() for p in model_path.iterdir())):
            return False, None

        for model_folder in model_path.iterdir():
            # 3) Each model folder must contain only one model definition file or folder.
            if sum(1 for _ in model_folder.iterdir()) != 1:
                logger.warning(
                    f"Model repository {model_folder!r} contains more than one model definition file or folder "
                    "so not treated as NamedModel."
                )
                return False, None

        return True, cls.model_type
