# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Optional, Tuple, Type, Union

from .model import Model


class ModelFactory:
    """ModelFactory is a class that provides a way to create a model object."""

    @staticmethod
    def create(path: Union[str, Path], name: str = "", model_type: str = "") -> Optional[Model]:
        """Creates a model object.

        Args:
            path (Union[str, Path]): A path to the model.
            name (str): A name of the model.
            model_type (str): A type of the model.

        Returns:
            A model object. Returns None if the model file/folder does not exist.
        """
        model_type, model_cls = ModelFactory.detect_model_type(path, model_type)

        if model_type and model_cls:
            model = model_cls(str(path), name)
            return model
        else:
            return None

    @staticmethod
    def detect_model_type(path: Union[str, Path], model_type: str = "") -> Tuple[str, Optional[Type[Model]]]:
        """Detects the model type based on a model path.

        Args:
            path (Union[str, Path]): A path to the model file/folder.
            model_type (str): A model type.

        Returns:
            A tuple of the model type string and the model class.
        """
        path = Path(path)

        for model_cls in Model.registered_models():
            # If a model_type is specified, check if it matches the model type.
            if model_type and model_cls.model_type != model_type:
                continue

            accept, model_type = model_cls.accept(path)
            if accept:
                return model_type, model_cls

        return "", None
