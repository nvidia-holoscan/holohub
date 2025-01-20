import logging
from pathlib import Path

from holoscan.conditions import CountCondition
from holoscan.core import Application
from operators.medical_imaging.core.app_context import AppContext
from operators.medical_imaging.operators.nii_data_loader_operator import NiftiDataLoader


class AIHFApp(Application):
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
        app_context: AppContext = Application.init_app_context(self.argv)
        self._logger.debug(f"Begin {self.compose.__name__}")
        app_input_path = Path(app_context.input_path)
        app_output_path = Path(app_context.output_path).resolve()

        # Parse hugging face pipeline info:
        # 1. If using a built-in pipeline, just need to provide the pipeline name
        # 2. If using a customized pipeline, either provide the hub uri or the local path
        pipeline_type = app_context.pipeline_type
        pipeline_name = app_context.pipeline_name
        pipeline_uri = app_context.pipeline_uri

        nii_data_loader = NiftiDataLoader(self, CountCondition(self,1), input_path=app_input_path)

        