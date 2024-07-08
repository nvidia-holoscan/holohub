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

from .model import Model


class TritonModel(Model):
    """Represents Triton models in the model repository.

    Triton Inference Server models are stored in a directory structure like this
    (https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md):

    ::

        <model-repository-path>/
            <model-name>/
            [config.pbtxt]
            [<output-labels-file> ...]
            <version>/
                <model-definition-file>
            <version>/
                <model-definition-file>
            ...
            <model-name>/
            [config.pbtxt]
            [<output-labels-file> ...]
            <version>/
                <model-definition-file>
            <version>/
                <model-definition-file>
            ...
            ...

    This class checks if the given path meets the folder structure of Triton:

    1) The path should be a folder path.

    2) The directory should contain only sub folders (model folders).

    3) Each model folder must contain a config.pbtxt file.

       a. A config.pbtxt file may contain model name.
          In that case, model's name should match with the folder name.

    4) Each model folder must include one or more folders having a positive integer value as name.

       a. Each such folder must contain a folder or file whose file name (without extension) is 'model'.

    It currently doesn't identify which model version would be selected.
    Model items identified would have a folder path, not a specific model file path.
    """

    model_type: str = "triton"

    def __init__(self, path: str, name: str = ""):
        """Initializes a TritonModel.

        This assumes that the given path is a valid Triton model repository.

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
                self._items[model_folder.name] = Model(str(model_folder), model_folder.name)

    @classmethod
    def accept(cls, path: str):
        model_path: Path = Path(path)

        # 1) The path should be a folder path.
        if not model_path.is_dir():
            return False, None

        # 2) The directory should contain only sub folders (model folders).
        if not all((p.is_dir() for p in model_path.iterdir())):
            return False, None

        is_triton_model_repository = True
        for model_folder in model_path.iterdir():
            # 3) Each model folder must contain a config.pbtxt file.
            if not (model_folder / "config.pbtxt").exists():
                return False, None
            # TODO(gigony): We do not check if the config.pbtxt file contains model name for now (3-1).
            # We assume that the model name is the same as the folder name.

            # 4) Each model folder must include one or more folders having a positive integer value as name.
            found_model = False
            for version_folder in model_folder.iterdir():
                version_folder_name = version_folder.name
                if version_folder.is_dir() and version_folder_name.isnumeric() and int(version_folder_name) > 0:
                    # 4-1) Each such folder must contain a folder or file whose file name (without extension)
                    #      is 'model'.
                    # TODO(gigony): check config.pbtxt file to see actual model file if specified.
                    if any(version_folder.glob("model.*")):
                        found_model = True
                    else:
                        return False, None
            if not found_model:
                is_triton_model_repository = False
                break
        if is_triton_model_repository:
            return True, cls.model_type

        return False, None
