#!/bin/bash
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

import os.path
from pathlib import Path
from typing import Any, Dict, ItemsView, List, Tuple

from operators.medical_imaging.exceptions import ItemNotExistsError, UnknownTypeError

# Store all supported model types in the order they should be checked
REGISTERED_MODELS = []


class Model:
    """Represents a model or a model repository.

    This encapsulates model's name and path.

    If this presents a model repository, repository's name and path are accessed via 'name' and 'path' attributes.

    If this presents a model, the model's name and path are accessed via 'name' and 'path' attributes.

    If the model's path is not specified(`Model("")`), the model is considered as a null model
    and `bool(Model("")) == False`.

    All models that this class represents can be retrieved by using `items()` method and a model with specific name
    can be retrieved by `get()` method with a model name argument (If only one model is available, you can skip
    specifying the model name).

    Loaded model object can be accessed via 'predictor' attribute and the predictor can be called
    using `__call__` method.

    In the `Operator` class, A model is accessible via `context.models` attribute inside `compute` method.
    Some subclasses (such as TorchModel) loads model file when `predictor` attribute is accessed so you can
    call(`__call__`) the model directly.

    >>> class MyOperator(Operator):
    >>>     def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
    >>>         model = context.models.get()
    >>>         result = model(op_input.get().asnumpy())

    If you want to load a model file manually, please set 'predictor' attribute to a loaded model object.

    >>> class MyOperator(Operator):
    >>>     def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
    >>>         import torch
    >>>         model = context.models.get()
    >>>         model.predictor = torch.jit.load(model.path, map_location="cpu").eval()
    >>>         result = model(op_input.get().asnumpy())

    Supported model types can be registered using static 'register' method.
    """

    model_type: str = "generic"

    def __init__(self, path: str, name: str = ""):
        """Constructor of a model.

        If name is not provided, the model name is taken from the path.
        `_predicator` is set to None and it is expected to be set by the child class when needed.
        `_items` is set to an dictionary having itself ({self.name: self}) and it is expected to be cleared
        by the child class if the path presents a model repository.

        Args:
            path (str): A path to a model.
            name (str): A name of the model.
        """

        self._path = path

        if name:
            self._name = name
        else:
            self._name = Path(path).stem

        self._predictor = None

        # Add self to the list of models
        self._items: Dict[str, Model] = {self.name: self}

    @property
    def predictor(self):
        """Return a predictor of the model.

        Returns:
            A predictor of the model.
        """
        return self._predictor

    @predictor.setter
    def predictor(self, predictor: Any):
        """Set a predictor of the model.

        Args:
            predictor: A predictor of the model.
        """
        self._predictor = predictor

    @property
    def path(self):
        """Return a path to the model."""
        return self._path

    @property
    def name(self):
        """Return a name of the model."""
        return self._name

    @classmethod
    def class_name(cls):
        """Return a name of the model class."""
        return cls.__name__

    @staticmethod
    def register(cls_list):
        """Register a list of model classes."""
        global REGISTERED_MODELS
        REGISTERED_MODELS = cls_list

    @staticmethod
    def registered_models():
        """Return a list of registered model classes."""
        return REGISTERED_MODELS

    @classmethod
    def accept(cls, path: str) -> Tuple[bool, str]:
        """Check if the path is a type of this model class.

        Args:
            path (str): A path to a model.

        Returns:
            (True, <model_type>) if the path is a type of this model class, (False, "") otherwise.
        """
        if not os.path.exists(path):
            return False, ""
        return True, cls.model_type

    def get(self, name: str = "") -> "Model":
        """Return a model object by name.

        If there is only one model in the repository or the model path, model object can be returned without specifying
        name.

        If there are more than one models in the repository, the model object can be returned by name whose name
        matches the provided name.

        Args:
            name (str): A name of the model.

        Returns:
            A model object is returned, matching the provided name if given.
        """
        if name:
            item = self._items.get(name)
            if item:
                return item
            else:
                raise ItemNotExistsError(f"A model with {name!r} does not exist.")
        else:
            item_count = len(self._items)
            if item_count == 1:
                return next(iter(self._items.values()))
            elif item_count > 1:
                raise UnknownTypeError(
                    f"There are more than one model. It should be one of ({', '.join(self._items.keys())})."
                )
            else:
                return self

    def get_model_list(self) -> List[Dict[str, str]]:
        """Return a list of models in the repository.

        If this model represents a model repository, then a list of model objects (name and path) is returned.
        Otherwise, a single model object list is returned.

        Returns:
            A list of models (name, path dictionary) in the repository.
        """
        model_list = []
        model_items = self.items()

        for _, m in model_items:
            model_list.append({"name": m.name, "path": os.path.abspath(m.path)})

        return model_list

    def items(self) -> ItemsView[str, "Model"]:
        """Return an ItemsView of models that this Model instance has.

        If this model represents a model repository, then an ItemsView of submodel objects is returned.
        Otherwise, an ItemsView of a single model object (self) is returned.

        Returns:
            An ItemView of models: `<model name>: <model object>`.
        """
        return self._items.items()

    def __call__(self, *args, **kwargs) -> Any:
        """Return a call of predictor of the model.

        Args:
            *args: A list of positional arguments.
            **kwargs: A dictionary of keyword arguments.

        Returns:
            A call of predictor of the model.

        Exceptions:
            ItemNotExistsError: If the predictor(model) is not set.
        """
        if self.predictor:
            return self.predictor(*args, **kwargs)
        else:
            raise ItemNotExistsError("A predictor of the model is not set.")

    def __bool__(self):
        """Return True if the model path is specified."""
        return bool(self.path)
