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

import os.path
from typing import Any
from zipfile import BadZipFile, ZipFile

from operators.medical_imaging.utils.importutil import optional_import

from .model import Model

torch, _ = optional_import("torch")


class TorchScriptModel(Model):
    """Represents TorchScript model.

    TorchScript serialization format (TorchScript model file) is created by torch.jit.save() method and
    the serialized model (which usually has .pt or .pth extension) is a ZIP archive containing many files.
    (https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md)

    We consider that the model is a torchscript model if its unzipped archive contains files named 'data.pkl' and
    'constants.pkl', and folders named 'code' and 'data'.

    When predictor property is accessed or the object is called (__call__), the model is loaded in `evaluation mode`
    from the serialized model file (if it is not loaded yet) and the model is ready to be used.
    """

    model_type: str = "torch_script"

    @property
    def predictor(self) -> "torch.nn.Module":  # type: ignore
        """Get the model's predictor (torch.nn.Module)

        If the predictor is not loaded, load it from the model file in evaluation mode.
        (https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html?highlight=eval#torch.jit.ScriptModule.eval)

        Returns:
            torch.nn.Module: the model's predictor
        """
        if self._predictor is None:
            # Use a device to dynamically remap, depending on the GPU availability.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._predictor = torch.jit.load(self.path, map_location=device).eval()
        return self._predictor

    @predictor.setter
    def predictor(self, predictor: Any):
        self._predictor = predictor

    def eval(self) -> "TorchScriptModel":
        """Set the model in evaluation model.

        This is a proxy method for torch.jit.ScriptModule.eval().
        See https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html?highlight=eval#torch.jit.ScriptModule.eval

        Returns:
            self
        """
        self.predictor.eval()
        return self

    def train(self, mode: bool = True) -> "TorchScriptModel":
        """Set the model in training mode.

        This is a proxy method for torch.jit.ScriptModule.train().
        See https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html?highlight=train#torch.jit.ScriptModule.train

        Args:
            mode (bool): whether the model is in training mode

        Returns:
            self
        """
        self.predictor.train(mode)
        return self

    @classmethod
    def accept(cls, path: str):
        prefix_code = False
        prefix_data = False
        prefix_constants_pkl = False
        prefix_data_pkl = False

        if not os.path.isfile(path):
            return False, None

        try:
            zip_file = ZipFile(path)
            for data in zip_file.filelist:
                file_name = data.filename
                pivot = file_name.find("/")
                if pivot != -1 and not prefix_code and file_name[pivot:].startswith("/code/"):
                    prefix_code = True
                if pivot != -1 and not prefix_data and file_name[pivot:].startswith("/data/"):
                    prefix_data = True
                if (
                    pivot != -1
                    and not prefix_constants_pkl
                    and file_name[pivot:] == "/constants.pkl"
                ):
                    prefix_constants_pkl = True
                if pivot != -1 and not prefix_data_pkl and file_name[pivot:] == "/data.pkl":
                    prefix_data_pkl = True
        except BadZipFile:
            return False, None

        if prefix_code and prefix_data and prefix_constants_pkl and prefix_data_pkl:
            return True, cls.model_type

        return False, None
