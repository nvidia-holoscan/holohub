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

import json
import logging
import os
import pickle
import tempfile
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from holoscan.core import Fragment, OperatorSpec

from operators.medical_imaging.core import AppContext, Image, IOType
from operators.medical_imaging.inference_operator import InferenceOperator
from operators.medical_imaging.utils.importutil import optional_import

MONAI_UTILS = "monai.utils"
nibabel, _ = optional_import("nibabel", "3.2.1")
torch, _ = optional_import("torch", "1.10.2")

NdarrayOrTensor, _ = optional_import("monai.config", name="NdarrayOrTensor")
MetaTensor, _ = optional_import("monai.data.meta_tensor", name="MetaTensor")
PostFix, _ = optional_import(
    "monai.utils.enums", name="PostFix"
)  # For the default meta_key_postfix
first, _ = optional_import("monai.utils.misc", name="first")
ensure_tuple, _ = optional_import(MONAI_UTILS, name="ensure_tuple")
convert_to_dst_type, _ = optional_import(MONAI_UTILS, name="convert_to_dst_type")
Key, _ = optional_import(MONAI_UTILS, name="ImageMetaKey")
MetaKeys, _ = optional_import(MONAI_UTILS, name="MetaKeys")
SpaceKeys, _ = optional_import(MONAI_UTILS, name="SpaceKeys")
Compose_, _ = optional_import("monai.transforms", name="Compose")
ConfigParser_, _ = optional_import("monai.bundle", name="ConfigParser")
MapTransform_, _ = optional_import("monai.transforms", name="MapTransform")
SimpleInferer, _ = optional_import("monai.inferers", name="SimpleInferer")

# Dynamic class is not handled so make it Any for now: https://github.com/python/mypy/issues/2477
Compose: Any = Compose_
MapTransform: Any = MapTransform_
ConfigParser: Any = ConfigParser_


__all__ = ["MonaiBundleInferenceOperator", "IOMapping", "BundleConfigNames"]


def get_bundle_config(bundle_path, config_names):
    """
    Gets the configuration parser from the specified Torchscript bundle file path.
    """

    bundle_suffixes = (".json", ".yaml", "yml")  # The only supported file ext(s)
    config_folder = "extra"

    def _read_from_archive(archive, root_name: str, config_name: str, do_search=True):
        """A helper function for reading the content of a config in the zip archive.

        Tries to read config content at the expected path in the archive, if error occurs,
        search and read with alternative paths.
        """

        content_text = None
        config_name = config_name.split(".")[0]  # In case ext is present

        # Try directly read with constructed and expected path into the archive
        for suffix in bundle_suffixes:
            path = Path(root_name, config_folder, config_name).with_suffix(suffix)
            try:
                logging.debug(f"Trying to read config {config_name!r} content from {path!r}.")
                content_text = archive.read(str(path))
                break
            except Exception:
                logging.debug(f"Error reading from {path}. Will try alternative ways.")
                continue

        # Try search for the name in the name list of the archive
        if not content_text and do_search:
            logging.debug(f"Trying to find the file in the archive for config {config_name!r}.")
            name_list = archive.namelist()
            for suffix in bundle_suffixes:
                for n in name_list:
                    if (f"{config_name}{suffix}").casefold in n.casefold():
                        logging.debug(
                            f"Trying to read content of config {config_name!r} from {n!r}."
                        )
                        content_text = archive.read(n)
                        break

        if not content_text:
            raise IOError(
                f"Cannot read config {config_name}{bundle_suffixes} or its content in the archive."
            )

        return content_text

    def _extract_from_archive(
        archive,
        root_name: str,
        config_names: List[str],
        dest_folder: Union[str, Path],
        do_search=True,
    ):
        """A helper function for extract files of configs from the archive to the destination folder

        Tries to extract with the full paths from the archive file, if error occurs, tries to search for
        and read from the file(s) if do_search is true.
        """

        config_names = [cn.split(".")[0] for cn in config_names]  # In case the extension is present
        file_list = []

        # Try directly read first with path into the archive
        for suffix in bundle_suffixes:
            try:
                logging.debug(f"Trying to extract {config_names} with ext {suffix}.")
                file_list = [
                    str(Path(root_name, config_folder, cn).with_suffix(suffix))
                    for cn in config_names
                ]
                archive.extractall(members=file_list, path=dest_folder)
                break
            except Exception as ex:
                file_list = []
                logging.debug(
                    f"Will try file search after error on extracting {config_names} with {file_list}: {ex}"
                )
                continue

        # If files not extracted, try search for expected files in the name list of the archive
        if (len(file_list) < 1) and do_search:
            logging.debug(f"Trying to find the config files in the archive for {config_names}.")
            name_list = archive.namelist()
            leftovers = deepcopy(config_names)  # to track any that are not found.
            for cn in config_names:
                for suffix in bundle_suffixes:
                    found = False
                    for n in name_list:
                        if (f"{cn}{suffix}").casefold() in n.casefold():
                            found = True
                            archive.extract(member=n, path=dest_folder)
                            break
                    if found:
                        leftovers.remove(cn)
                        break

            if len(leftovers) > 0:
                raise IOError(f"Failed to extract content for these config(s): {leftovers}.")

        return file_list

    # End of helper functions

    if isinstance(config_names, str):
        config_names = [config_names]

    name, _ = os.path.splitext(
        os.path.basename(bundle_path)
    )  # bundle file name same archive folder name
    parser = ConfigParser()

    # Parser to read the required metadata and extra config contents from the archive
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(bundle_path, "r") as archive:
            metadata_config_name = "metadata"
            metadata_text = _read_from_archive(archive, name, metadata_config_name)
            parser.read_meta(f=json.loads(metadata_text))

            # now get the other named configs
            file_list = _extract_from_archive(archive, name, config_names, tmp_dir)
            parser.read_config([Path(tmp_dir, f_path) for f_path in file_list])

    parser.parse()

    return parser


DISALLOW_LOAD_SAVE = ["LoadImage", "SaveImage"]
DISALLOW_SAVE = ["SaveImage"]


def filter_compose(compose, disallowed_prefixes):
    """
    Removes transforms from the given Compose object whose names begin with `disallowed_prefixes`.
    """
    filtered = []
    for t in compose.transforms:
        tname = type(t).__name__
        if not any(dis in tname for dis in disallowed_prefixes):
            filtered.append(t)

    compose.transforms = tuple(filtered)
    return compose


def is_map_compose(compose):
    """
    Returns True if the given Compose object uses MapTransform instances.
    """
    return isinstance(first(compose.transforms), MapTransform)


class IOMapping:
    """This object holds an I/O definition for an operator."""

    def __init__(
        self,
        label: str,
        data_type: Type,
        storage_type: IOType,
    ):
        """Creates an object holding an operator I/O definitions.

        Limitations apply with the combination of data_type and storage_type, which will
        be validated at runtime.

        Args:
            label (str): Label for the operator input or output.
            data_type (Type): Datatype of the I/O data content.
            storage_type (IOType): The storage type expected, i.e. IN_MEMORY or DISK.
        """
        self.label: str = label
        self.data_type: Type = data_type
        self.storage_type: IOType = storage_type


class BundleConfigNames:
    """This object holds the name of relevant config items used in a MONAI Bundle."""

    def __init__(
        self,
        preproc_name: str = "preprocessing",
        postproc_name: str = "postprocessing",
        inferer_name: str = "inferer",
        config_names: Union[List[str], Tuple[str], str] = "inference",
    ) -> None:
        """Creates an object holding the names of relevant config items in a MONAI Bundle.

        This object holds the names of the config items in a MONAI Bundle that will need to be
        parsed by the inference operator for automating the object creations and inference.
        Defaults values are provided per conversion, so the arguments only need to be set as needed.

        Args:
            preproc_name (str, optional): Name of the config item for pre-processing transforms.
                                          Defaults to "preprocessing".
            postproc_name (str, optional): Name of the config item for post-processing transforms.
                                           Defaults to "postprocessing".
            inferer_name (str, optional): Name of the config item for inferer.
                                          Defaults to "inferer".
            config_names (List[str], optional): Name of config file(s) in the Bundle for parsing.
                                                Defaults to ["inference"]. File ext must be .json.
        """

        def _ensure_str_list(config_names):
            names = []
            if isinstance(config_names, (List, Tuple)):
                if len(config_names) < 1:
                    raise ValueError("At least one config name must be provided.")
                names = [str(name) for name in config_names]
            else:
                names = [str(config_names)]

            return names

        self.preproc_name: str = preproc_name
        self.postproc_name: str = postproc_name
        self.inferer_name: str = inferer_name
        self.config_names: List[str] = _ensure_str_list(config_names)


DEFAULT_BundleConfigNames = BundleConfigNames()


# The operator env decorator defines the required pip packages commonly used in the Bundles.
# The MONAI Deploy App SDK packager currently relies on the App to consolidate all required packages in order to
# install them in the MAP Docker image.
# TODO: Dynamically setting the pip_packages env on init requires the bundle path be passed in. Apps using this
#       operator may choose to pass in a accessible bundle path at development and packaging stage. Ideally,
#       the bundle path should be passed in by the Packager, e.g. via env var, when the App is initialized.
#       As of now, the Packager only passes in the model path after the App including all operators are init'ed.
# @md.env(pip_packages=["monai>=1.0.0", "torch>=1.10.02", "numpy>=1.21", "nibabel>=3.2.1"])
class MonaiBundleInferenceOperator(InferenceOperator):
    """This inference operator automates the inference operation for a given MONAI Bundle.

    This inference operator configures itself based on the parsed data from a MONAI bundle file. This file is included
    with a MAP as a Torchscript file with added bundle metadata or a zipped bundle with weights. The class will
    configure how to do pre- and post-processing, inference, which device to use, state its inputs, outputs, and
    dependencies. Its compute method is meant to be general purpose to most any bundle such that it will handle
    any input specified in the bundle and produce output as specified, using the inference object the bundle defines.
    A number of methods are provided which define parts of functionality relating to this behavior, users may wish
    to overwrite these to change behavior is needed for specific bundles.

    The input(s) and output(s) for this operator need to be provided when an instance is created, and their labels need
    to correspond to the bundle network input and output names, which are also used as the keys in the pre and post processing.

    For image input and output, the type is the `Image` class. For output of probabilities, the type is `Dict`.

    This operator is expected to be linked with both source and destination operators, e.g. receiving an `Image` object from
    the `DICOMSeriesToVolumeOperator`, and passing a segmentation `Image` to the `DICOMSegmentationWriterOperator`.
    In such cases, the I/O storage type can only be `IN_MEMORY` due to the restrictions imposed by the application executor.

    For the time being, the input and output to this operator are limited to in_memory object.
    """

    known_io_data_types = {
        "image": Image,  # Image object
        "series": np.ndarray,
        "tuples": np.ndarray,
        "probabilities": Dict[str, Any],  # dictionary containing probabilities and predicted labels
    }

    kw_preprocessed_inputs = "preprocessed_inputs"

    # For testing the app directly, the model should be at the following path.
    MODEL_LOCAL_PATH = Path(os.environ.get("HOLOSCAN_MODEL_PATH", Path.cwd() / "model/model.ts"))

    def __init__(
        self,
        fragment: Fragment,
        *args,
        app_context: AppContext,
        input_mapping: List[IOMapping],
        output_mapping: List[IOMapping],
        model_name: Optional[str] = "",
        bundle_path: Optional[Union[Path, str]] = None,
        bundle_config_names: Optional[BundleConfigNames] = DEFAULT_BundleConfigNames,
        **kwargs,
    ):
        """Create an instance of this class, associated with an Application/Fragment.

        Args:
            fragment (Fragment): An instance of the Application class which is derived from Fragment.
            app_context (AppContext): Object holding the I/O and model paths, and potentially loaded models.
            input_mapping (List[IOMapping]): Define the inputs' name, type, and storage type.
            output_mapping (List[IOMapping]): Defines the outputs' name, type, and storage type.
            model_name (Optional[str], optional): Name of the model/bundle, needed in multi-model case.
                                                  Defaults to "".
            bundle_path (Optional[str], optional): Known path to the bundle file. Defaults to None.
            bundle_config_names (BundleConfigNames, optional): Relevant config item names in a the bundle.
                                                               Defaults to DEFAULT_BundleConfigNames.
        """

        self._executing = False
        self._lock = Lock()

        self._model_name = model_name.strip() if isinstance(model_name, str) else ""
        self._bundle_config_names = (
            bundle_config_names if bundle_config_names else BundleConfigNames()
        )
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping

        self._parser: ConfigParser = (
            None  # Needs known bundle path, either on init or when compute function is called.
        )
        self._inferer: Any = None  # Will be set during bundle parsing.
        self._init_completed: bool = False

        # Need to set the operator's input(s) and output(s). Even when the bundle parsing is done in init,
        # there is still a need to define what op inputs/outputs map to what keys in the bundle config,
        # along with the op input/output storage type.
        # Also, the App Executor needs to set the IO context of the operator before calling the compute function.
        # Delay till setup is called, as the Application object does support the add_input and add_output now.
        # self._add_inputs(self._input_mapping)
        # self._add_outputs(self._output_mapping)

        # Complete the init if the bundle path is known, otherwise delay till the compute function is called
        # and try to get the model/bundle path from the execution context.
        try:
            self._bundle_path = (
                Path(bundle_path) if bundle_path and len(str(bundle_path).strip()) > 0 else None
            )

            if self._bundle_path and self._bundle_path.is_file():
                self._init_config(self._bundle_config_names.config_names)
                self._init_completed = True
            else:
                logging.debug(
                    f"Bundle, at path {self._bundle_path}, not available. Will get it in the execution context."
                )
                self._bundle_path = None
        except Exception:
            logging.warn(
                "Bundle parsing is not completed on init, delayed till this operator is called to execute."
            )
            self._bundle_path = None

        self._fragment = fragment  # In case it is needed.
        self.app_context = app_context

        # Lazy init of model network till execution time when the context is fully set.
        self._model_network: Any = None

        super().__init__(fragment, *args, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        if not name or isinstance(name, str):
            raise ValueError(f"Value, {name}, must be a non-empty string.")
        self._model_name = name

    @property
    def bundle_path(self) -> Union[Path, None]:
        """The path of the MONAI Bundle model."""
        return self._bundle_path

    @bundle_path.setter
    def bundle_path(self, bundle_path: Union[str, Path]):
        if not bundle_path or not Path(bundle_path).expanduser().is_file():
            raise ValueError(f"Value, {bundle_path}, is not a valid file path.")
        self._bundle_path = Path(bundle_path).expanduser().resolve()

    @property
    def parser(self) -> Union[ConfigParser, None]:
        """The ConfigParser object."""
        return self._parser

    @parser.setter
    def parser(self, parser: ConfigParser):
        if parser and isinstance(parser, ConfigParser):
            self._parser = parser
        else:
            raise ValueError("Value must be a valid ConfigParser object.")

    def _init_config(self, config_names):
        """Completes the init with a known path to the MONAI Bundle

        Args:
            config_names ([str]): Names of the config (files) in the bundle
        """

        parser = get_bundle_config(str(self._bundle_path), config_names)
        self._parser = parser

        meta = self.parser["_meta_"]

        # When this function is NOT called by the __init__, setting the pip_packages env here
        # will not get dependencies to the App SDK Packager to install the packages in the MAP.
        # pip_packages = ["monai"] + [f"{k}=={v}" for k, v in meta["optional_packages_version"].items()]

        # Currently not support adding and installing dependent pip package at runtime.
        # if self._env:
        #     self._env.pip_packages.extend(pip_packages)  # Duplicates will be figured out on use.
        # else:
        #     self._env = OperatorEnv(pip_packages=pip_packages)

        if parser.get("device") is not None:
            self._device = parser.get_parsed_content("device")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if parser.get(self._bundle_config_names.inferer_name) is not None:
            self._inferer = parser.get_parsed_content(self._bundle_config_names.inferer_name)
        else:
            self._inferer = SimpleInferer()

        self._inputs = meta["network_data_format"]["inputs"]
        self._outputs = meta["network_data_format"]["outputs"]

        # Given the restriction on operator I/O storage type, and known use cases, the I/O storage type of
        # this operator is limited to IN_MEMRORY objects, so we will remove the LoadImage and SaveImage
        self._preproc = self._get_compose(
            self._bundle_config_names.preproc_name, DISALLOW_LOAD_SAVE
        )
        self._postproc = self._get_compose(
            self._bundle_config_names.postproc_name, DISALLOW_LOAD_SAVE
        )

        # Need to find out the meta_key_postfix. The key name of the input concatenated with this postfix
        # will be the key name for the metadata for the input.
        # Customized metadata key names are not supported as of now.
        self._meta_key_postfix = self._get_meta_key_postfix(self._preproc)

        logging.debug(
            f"Effective transforms in pre-processing: {[type(t).__name__ for t in self._preproc.transforms]}"
        )
        logging.debug(
            f"Effective Transforms in post-processing: {[type(t).__name__ for t in self._preproc.transforms]}"
        )

    def _get_compose(self, obj_name, disallowed_prefixes):
        """Gets a Compose object containing a sequence of transforms from item `obj_name` in `self._parser`."""

        if self._parser.get(obj_name) is not None:
            compose = self._parser.get_parsed_content(obj_name)
            return filter_compose(compose, disallowed_prefixes)

        return Compose([])

    def _get_meta_key_postfix(self, compose: Compose, key_name: str = "meta_key_postfix") -> str:
        post_fix = PostFix.meta()
        if compose and key_name:
            for t in compose.transforms:
                if isinstance(t, MapTransform) and hasattr(t, key_name):
                    post_fix = getattr(t, key_name)
                    # For some reason the attr is a tuple
                    if isinstance(post_fix, tuple):
                        post_fix = str(post_fix[0])
                    break

        return str(post_fix)

    def _get_io_data_type(self, conf):
        """
        Gets the input/output type of the given input or output metadata dictionary. The known Python types for input
        or output types are given in the dictionary `BundleOperator.known_io_data_types` which relate type names to
        the actual type. if `conf["type"]` is an actual object that's not a string then this is assumed to be the
        type specifier and is returned. The fallback type is `bytes` which indicates the value is a pickled object.

        Args:
            conf: configuration dictionary for an input or output from the "network_data_format" metadata section

        Returns:
            A concrete type associated with this input/output type, this can be Image or np.ndarray or a Python type
        """

        # The Bundle's network_data_format for inputs and outputs does not indicate the storage type, i.e. IN_MEMORY
        # or DISK, for the input(s) and output(s) of the operators. Configuration is required, though limited to
        # IN_MEMORY for now.
        # Certain association and transform are also required. The App SDK IN_MEMORY I/O can hold
        # Any type, so if the type match and content format matches, data can simply be used as is, however, with
        # the Type being Image, the object needs to be converted before being used as the expected "image" type.
        ctype = conf["type"]
        if ctype in self.known_io_data_types:  # known type name from the specification
            return self.known_io_data_types[ctype]
        elif isinstance(ctype, type):  # type object
            return ctype
        else:  # don't know, something that hasn't been figured out
            logging.warn(
                f"I/O data type, {ctype}, is not a known/supported type. Return as Type object."
            )
            return object

    def _add_inputs(self, input_mapping: List[IOMapping]):
        """Adds operator inputs as specified."""

        [self.add_input(v.label, v.data_type, v.storage_type) for v in input_mapping]

    def _add_outputs(self, output_mapping: List[IOMapping]):
        """Adds operator outputs as specified."""

        [self.add_output(v.label, v.data_type, v.storage_type) for v in output_mapping]

    def setup(self, spec: OperatorSpec):
        [spec.input(v.label) for v in self._input_mapping]
        for v in self._output_mapping:
            if (
                v.storage_type == IOType.IN_MEMORY
            ):  # As of now the output port type can only be in_memory object.
                spec.output(v.label)

    def compute(self, op_input, op_output, context):
        """Infers with the input(s) and saves the prediction result(s) to output

        Args:
            op_input (InputContext): An input context for the operator.
            op_output (OutputContext): An output context for the operator.
            context (ExecutionContext): An execution context for the operator.
        """

        # Try to get the Model object and its path from the context.
        #   If operator is not fully initialized, use model path as bundle path to finish it.
        # If Model not loaded, but bundle path exists, load model; edge case for local dev.
        #
        # `context.models.get(model_name)` returns a model instance if exists.
        # If model_name is not specified and only one model exists, it returns that model.

        # The models are loaded on construction via the AppContext object in turn the model factory.
        self._model_network = (
            self.app_context.models.get(self._model_name) if self.app_context.models else None
        )

        if self._model_network:
            if not self._init_completed:
                with self._lock:
                    if not self._init_completed:
                        self._bundle_path = self._model_network.path
                        logging.info(f"Parsing from bundle_path: {self._bundle_path}")
                        self._init_config(self._bundle_config_names.config_names)
                        self._init_completed = True
        elif self._bundle_path:
            # For the case of local dev/testing when the bundle path is not passed in as an exec cmd arg.
            # When run as a MAP docker, the bundle file is expected to be in the context, even if the model
            # network is loaded on a remote inference server (when the feature is introduced).
            logging.debug(
                f"Model network not loaded. Trying to load from model path: {self._bundle_path}"
            )
            self._model_network = torch.jit.load(self.bundle_path, map_location=self._device).eval()
        else:
            raise IOError("Model network is not load and model file not found.")

        first_input_name, *other_names = list(self._inputs.keys())

        with torch.no_grad():
            inputs: Any = {}  # Use type Any to quiet MyPy type checking complaints.

            start = time.time()
            for name in self._inputs.keys():
                # Input MetaTensor creation is based on the same logic in monai LoadImage
                # value: NdarrayOrTensor  # MyPy complaints
                value, meta_data = self._receive_input(name, op_input, context)
                value = convert_to_dst_type(value, dst=value)[0]
                if not isinstance(meta_data, dict):
                    raise ValueError("`meta_data` must be a dict.")
                value = MetaTensor.ensure_torch_and_prune_meta(value, meta_data)
                inputs[name] = value
                # Named metadata dict not needed any more, as it is in the MetaTensor

            inputs = self.pre_process(inputs)
            first_input_v = inputs[first_input_name]  # keep a copy of value for later use
            first_input = inputs.pop(first_input_name)[None].to(self._device)

            # select other tensor inputs
            other_inputs = {
                k: v[None].to(self._device)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            # select other non-tensor inputs
            other_inputs.update(
                {k: inputs[k] for k in other_names if not isinstance(inputs[k], torch.Tensor)}
            )
            logging.debug(
                f"Ingest and Pre-processing elapsed time (seconds): {time.time() - start}"
            )

            start = time.time()
            outputs: Any = self.predict(
                data=first_input, **other_inputs
            )  # Use type Any to quiet MyPy complaints.
            logging.debug(f"Inference elapsed time (seconds): {time.time() - start}")

            # Note that the `inputs` are needed because the `invert` transform requires it. With metadata being
            # in the keyed MetaTensors of inputs, e.g. `image`, the whole inputs are needed.
            start = time.time()
            inputs[first_input_name] = first_input_v
            kw_args = {self.kw_preprocessed_inputs: inputs}
            outputs = self.post_process(ensure_tuple(outputs)[0], **kw_args)
            logging.debug(f"Post-processing elapsed time (seconds): {time.time() - start}")
        if isinstance(outputs, (tuple, list)):
            output_dict = dict(zip(self._outputs.keys(), outputs))
        elif not isinstance(outputs, dict):
            output_dict = {first(self._outputs.keys()): outputs}
        else:
            output_dict = outputs

        for name in self._outputs.keys():
            # Note that the input metadata needs to be passed.
            # Please see the comments in the called function for the reasons.
            self._send_output(output_dict[name], name, first_input_v.meta, op_output, context)

    def predict(
        self, data: Any, *args, **kwargs
    ) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Predicts output using the inferer."""

        return self._inferer(inputs=data, network=self._model_network, *args, **kwargs)

    def pre_process(
        self, data: Any, *args, **kwargs
    ) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Processes the input dictionary with the stored transform sequence `self._preproc`."""

        if is_map_compose(self._preproc):
            return self._preproc(data)
        return {k: self._preproc(v) for k, v in data.items()}

    def post_process(
        self, data: Any, *args, **kwargs
    ) -> Union[Image, Any, Tuple[Any, ...], Dict[Any, Any]]:
        """Processes the output list/dictionary with the stored transform sequence `self._postproc`.

        The "processed_inputs", in fact the metadata in it, need to be passed in so that the
        invertible transforms in the post processing can work properly.
        """

        # Expect the inputs be passed in so that the inversion can work.
        inputs = kwargs.get(self.kw_preprocessed_inputs, {})

        if is_map_compose(self._postproc):
            if isinstance(data, (list, tuple)):
                outputs_dict = dict(zip(data, self._outputs.keys()))
            elif not isinstance(data, dict):
                oname = first(self._outputs.keys())
                outputs_dict = {oname: data}
            else:
                outputs_dict = data

            # Need to add back the inputs including metadata as they are needed by the invert transform.
            outputs_dict.update(inputs)
            logging.debug(f"Effective output dict keys: {outputs_dict.keys()}")
            return self._postproc(outputs_dict)
        else:
            if isinstance(data, (list, tuple)):
                return list(map(self._postproc, data))

            return self._postproc(data)

    def _receive_input(self, name: str, op_input, context):
        """Extracts the input value for the given input name."""

        # The op_input can have the storage type of IN_MEMORY with the data type being Image or others,
        # as well as the other type of DISK with data type being DataPath.
        # The problem is, the op_input object does not have an attribute for the storage type, which
        # needs to be inferred from data type, with DataPath meaning DISK storage type. The file
        # content type may be interpreted from the bundle's network input type, but it is indirect
        # as the op_input is the input for processing transforms, not necessarily directly for the network.
        in_conf = self._inputs[name]
        itype = self._get_io_data_type(in_conf)
        value = op_input.receive(name)

        metadata = None
        if isinstance(value, Path):
            if not value.exists():
                raise ValueError(f"Input path, {value}, does not exist.")

            file_path = value / name
            # The named input can only be a folder as of now, but just in case things change.
            if value.is_file():
                file_path = value
            elif not file_path.exists() and value.is_dir():
                # Expect one and only one file exists for use.
                files = [f for f in value.glob("*") if f.is_file()]
                if len(files) != 1:
                    raise ValueError(f"Input path, {value}, should have one and only one file.")

                file_path = files[0]

            # Only Python pickle file and or numpy file are supported as of now.
            with open(file_path, "rb") as f:
                if itype == np.ndarray:
                    value = np.load(file_path, allow_pickle=True)
                else:
                    value = pickle.load(f)

        # Once extracted, the input data may be further processed depending on its actual type.
        if isinstance(value, Image):
            # Need to get the image ndarray as well as metadata
            value, metadata = self._convert_from_image(value)
            logging.debug(f"Shape of the converted input image: {value.shape}")
            logging.debug(f"Metadata of the converted input image: {metadata}")
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value).to(self._device)

        # else value is some other object from memory

        return value, metadata

    def _send_output(self, value: Any, name: str, metadata: Dict, op_output, context):
        """Send the given output value to the output context."""

        logging.debug(f"Setting output {name}")

        out_conf = self._outputs[name]
        otype = self._get_io_data_type(out_conf)

        if otype == Image:
            # The value must be torch.tensor or ndarray. Note also that by convention the image/tensor
            # out of the MONAI post processing is [CWHD] with dim for batch already squeezed out.
            # Prediction image, e.g. segmentation image, needs to have its dimensions
            # rearranged to fit the conventions used by Image class, i.e. [DHW], without channel dim.
            # Also, based on known use cases, e.g. prediction being seg image and the destination
            # operators expect the data type to be unit8, conversion needs to be done as well.
            # Metadata, such as pixel spacing and orientation, also needs to be set in the Image object,
            # which is why metadata is expected to be passed in.
            # TODO: Revisit when multi-channel images are supported.

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            elif not isinstance(value, np.ndarray):
                raise TypeError("arg 1 must be of type torch.Tensor or ndarray.")

            logging.debug(f"Output {name} numpy image shape: {value.shape}")
            result: Any = Image(
                np.swapaxes(np.squeeze(value, 0), 0, 2).astype(np.uint8), metadata=metadata
            )
            logging.debug(f"Converted Image shape: {result.asnumpy().shape}")
        elif otype == np.ndarray:
            result = np.asarray(value)
        elif out_conf["type"] == "probabilities":
            _, value_class = value.max(dim=0)
            prediction = [out_conf["channel_def"][str(int(v))] for v in value.flatten()]

            result = {"result": prediction, "probabilities": value.cpu().numpy()}
        elif isinstance(value, torch.Tensor):
            result = value.cpu().numpy()

        # The operator output currently has many limitation depending on if the operator is
        # a leaf node or not. The get method throws for non-leaf node, irrespective of storage type,
        # and for leaf node if the storage type is IN_MEMORY.
        try:
            op_output_config = op_output.get(name)
            if isinstance(op_output_config, Path):
                output_file = op_output_config / name
                output_file.parent.mkdir(exist_ok=True)
                # Save pickle file
                with open(output_file, "wb") as wf:
                    pickle.dump(result, wf)

                # Cannot (re)set/modify the op_output path to the actual file like below
                # op_output.set(str(output_file), name)
            else:
                op_output.emit(result, name)
        except Exception:
            # The following throws if the output storage type is DISK, but The OutputContext
            # currently does not expose the storage type. Try and let it throw if need be.
            op_output.emit(result, name)

    def _convert_from_image(self, img: Image) -> Tuple[np.ndarray, Dict]:
        """Converts the Image object to the expected numpy array with metadata dictionary.

        Args:
            img: A SDK Image object.
        """

        # The Image class provides a numpy array and a metadata dict without a defined set of keys.
        # In most scenarios, if not all, DICOM series is converted to Image by the
        # DICOMSeriesToVolumeOperator, but the generated metadata lacks the specifics keys expected
        # by the MONAI transforms. So there is need to convert the Image object.
        # Also, there is not a defined key to express the source or producer of an Image object, so,
        # one has to inspect certain keys, based on known conversion, to infer the producer.
        # An issues already exists for the improvement of the Image class.

        img_meta_dict: Dict = img.metadata()

        if (
            not img_meta_dict
            or ("spacing" in img_meta_dict and "original_affine" in img_meta_dict)
            or "row_pixel_spacing" not in img_meta_dict
        ):
            return img.asnumpy(), img_meta_dict
        else:
            return self._convert_from_image_dicom_source(img)

    def _convert_from_image_dicom_source(self, img: Image) -> Tuple[np.ndarray, Dict]:
        """Converts the Image object to the expected numpy array with metadata dictionary.

        Args:
            img: A SDK Image object converted from DICOM instances.
        """

        img_meta_dict: Dict = img.metadata()
        meta_dict = deepcopy(img_meta_dict)

        # The MONAI ImageReader, e.g. the ITKReader, arranges the image spatial dims in WHD,
        # so the "spacing" needs to be expressed in such an order too, as expected by the transforms.
        meta_dict["spacing"] = np.asarray(
            [
                img_meta_dict["row_pixel_spacing"],
                img_meta_dict["col_pixel_spacing"],
                img_meta_dict["depth_pixel_spacing"],
            ]
        )
        # Use defines MetaKeys directly
        meta_dict[MetaKeys.ORIGINAL_AFFINE] = np.asarray(
            img_meta_dict.get("nifti_affine_transform", None)
        )
        meta_dict[MetaKeys.AFFINE] = meta_dict[MetaKeys.ORIGINAL_AFFINE].copy()
        meta_dict[MetaKeys.SPACE] = SpaceKeys.LPS  # not using SpaceKeys.RAS or affine_lps_to_ras

        # Similarly the Image ndarray has dim order DHW, to be rearranged to WHD.
        # TODO: Need to revisit this once multi-channel image is supported and the Image class itself
        #       is enhanced to provide attributes or functions for channel and dim order details.
        converted_image = np.swapaxes(img.asnumpy(), 0, 2)

        # The spatial shape is then that of the converted image, in WHD
        meta_dict[MetaKeys.SPATIAL_SHAPE] = np.asarray(converted_image.shape)

        # Well, now channel for now.
        meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM] = "no_channel"

        return converted_image, meta_dict
