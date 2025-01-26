import logging
import os
from pathlib import Path

from holoscan.conditions import CountCondition
from holoscan.core import Application
from operators.medical_imaging.operators.hugging_face_operator import (
    HuggingFacePipelineInputAdaptor,
    HuggingFacePipelineOperator,
)
from operators.medical_imaging.operators.nii_data_loader_operator import (
    NiftiDataLoader,
    NiftiDataWriter,
)


class AIHFApp(Application):
    PIPELINE_TYPE = "customized_remote"
    PIPELINE_NAME = "MONAI/VISTA3D-HF"

    def __init__(self, *args, **kwargs):
        """Create hugging face pipeline instance"""
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

    def run(self, *args, **kwargs):
        """This method calls the base class to run. Can be omitted if simply calling through."""
        self._logger.info(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.info(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""
        # Use Commandline options over environment variables to init context.
        self._logger.debug(f"Begin {self.compose.__name__}")
        app_input_path = Path(os.environ.get("HOLOSCAN_INPUT_PATH", None))
        app_output_path = Path(os.environ.get("HOLOSCAN_OUTPUT_PATH", None))
        if not app_input_path or not app_output_path:
            raise AttributeError(
                f"The input and output path must be set through HOLOSCAN_INPUT_PATH and HOLOSCAN_OUTPUT_PATH environment variables."
            )

        # Parse hugging face pipeline info:
        # 1. If using a built-in pipeline, just need to provide the pipeline name
        # 2. If using a customized pipeline, either provide the hub uri or the local path
        pipeline_type = self.PIPELINE_TYPE
        pipeline_name = self.PIPELINE_NAME

        nii_data_loader = NiftiDataLoader(
            self, CountCondition(self, 1), use_monai=True, input_path=app_input_path
        )
        # hf_pipeline_input_adaptor = HuggingFacePipelineInputAdaptor(
        #     self,
        #     keys=[
        #         "image",
        #     ],
        # )
        hf_pipeline_operator = HuggingFacePipelineOperator(
            self,
            name="hf_pipeline_op",
            pipeline_name=pipeline_name,
            pipeline_type=pipeline_type,
            pipeline_init_kwargs={"device": "cuda:0", "save_output": False},
        )
        nii_data_saver = NiftiDataWriter(self, name="nii_writer_op", image_path=app_output_path, user_monai=True)
        # self.add_flow(nii_data_loader, hf_pipeline_input_adaptor, {("image", "image")})
        self.add_flow(nii_data_loader, hf_pipeline_operator, {("image", "input_dict")})
        self.add_flow(hf_pipeline_operator, nii_data_saver, {("output_dict", "input_dict")})


if __name__ == "__main__":
    logging.info(f"Begin {__name__}")

    AIHFApp().run()

    logging.info(f"End {__name__}")
