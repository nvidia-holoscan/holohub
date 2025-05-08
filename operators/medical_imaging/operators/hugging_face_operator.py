import logging
import sys
import os
from enum import Enum
from typing import Dict, Sequence

from holoscan.core import Fragment, Operator, OperatorSpec
from huggingface_hub import snapshot_download
from transformers import pipeline
from monai.data import MetaTensor


class HF_Pipeline_Type(str, Enum):
    built_in = "built-in"
    customized_local = "customized_local"
    customized_remote = "customized_remote"


class HuggingFacePipelineInputAdaptor(Operator):
    """This is an adaptor to wrap input to the format that fits HuggingFacePipelineOperator"""

    OUTPUT_NAME = "output_dict"

    def __init__(self, fragment: Fragment, *args, keys: Sequence[str], **kwargs):
        self.keys = keys
        if not self.keys:
            raise AttributeError("Please provide input keys.")

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for key in self.keys:
            spec.input(key)
        spec.output(self.OUTPUT_NAME)

    def compute(self, op_input, op_output, context):
        outputs: Dict = {}
        for key in self.keys:
            outputs[key] = op_input.receive(key)
        op_output.emit(outputs, self.OUTPUT_NAME)


class HuggingFacePipelineOperator(Operator):
    """This is an operator to run the hugging face pipeline."""

    INPUT_NAME = "input_dict"
    OUTPUT_NAME = "output_dict"

    def __init__(
        self,
        fragment: Fragment,
        *args,
        pipeline_type: str,
        pipeline_name: str,
        pipeline_uri: os.PathLike | None = None,
        revision: str | None = None,
        pipeline_init_kwargs: Dict = {},
        **kwargs,
    ):
        self.pipeline_type = pipeline_type
        if self.pipeline_type not in {item.value for item in HF_Pipeline_Type}:
            raise AttributeError(
                f"Invalid pipeline type, expecting {self.HUGGING_FACE_PIPELINE_TYPES}, but got {self.pipeline_type}"
            )

        if self.pipeline_type == HF_Pipeline_Type.built_in:
            logging.info(f"Using built-in transformer pipeline {pipeline_name}")
            self.pipeline = pipeline(pipeline_name, revision=revision, **pipeline_init_kwargs)

        self.pipeline_uri = ""
        if self.pipeline_type == HF_Pipeline_Type.customized_local:
            self.pipeline_uri = pipeline_uri

        if self.pipeline_type == HF_Pipeline_Type.customized_remote:
            tmp_model_dir = snapshot_download(pipeline_name, revision=revision)
            logging.info(f"Cached the hugging face model to {tmp_model_dir}")
            self.pipeline_uri = tmp_model_dir

        self.pipeline = self._load_local_pipeline(**pipeline_init_kwargs)

        super().__init__(fragment, *args, **kwargs)

    def _load_local_pipeline(self, **kwargs):
        """Load pipeline from local path."""
        pipeline_uri = getattr(self, "pipeline_uri")
        if not pipeline_uri:
            raise AttributeError(f"Illegal customized pipeline uri {pipeline_uri}.")

        if pipeline_uri and os.path.exists(pipeline_uri):
            current_dir = os.getcwd()
            os.chdir(pipeline_uri)
            sys.path.append(pipeline_uri)
            try:
                from hugging_face_pipeline import HuggingFacePipelineHelper
            except:
                raise FileNotFoundError(
                    "Cannot import hugging face pipeline, please check the provided pipeline."
                )
            sys.path.remove(pipeline_uri)
            os.chdir(current_dir)
            model_path = os.path.join(pipeline_uri, "vista3d_pretrained_model")
            hf_pipeline = HuggingFacePipelineHelper().init_pipeline(model_path, **kwargs)
            return hf_pipeline
        else:
            raise FileNotFoundError("Cannot find the hugging face pipeline files.")

    def setup(self, spec: OperatorSpec):
        spec.input(self.INPUT_NAME)
        spec.output(self.OUTPUT_NAME)

    def compute(self, op_input, op_output, context):
        input_dict = {}
        input_image = op_input.receive(self.INPUT_NAME)
        input_data = input_image.asnumpy()
        meta_data = input_image.metadata()
        input_dict = {"image": MetaTensor(input_data, affine=meta_data["affine"], meta=meta_data)}
        input_dict.update({"label_prompt": [3]})
        outputs = self.pipeline(input_dict)
        op_output.emit(outputs, self.OUTPUT_NAME)
