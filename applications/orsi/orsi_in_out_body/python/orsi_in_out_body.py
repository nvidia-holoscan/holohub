import os
from holoscan.core import Application
from holoscan.operators import (
    InferenceOp,
    VideoStreamReplayerOp,
)
# from holoscan.videomaster import VideoMasterSourceOp

from holoscan.resources import UnboundedAllocator
from holohub.orsi_format_converter import OrsiFormatConverterOp
from holohub.orsi_segmentation_preprocessor import OrsiSegmentationPreprocessorOp
from holohub.orsi_visualizer import OrsiVisualizationOp

class OrsiInOutBodyApp(Application):
    def __init__(self):
        super().__init__()
        self.name = "OrsiInOutBodyApp"
        self.data_path = "data/orsi"


    def compose(self):

        pool=UnboundedAllocator(self, name="pool")

        source_type = self.kwargs("source")['source']
        if source_type.lower() == "replayer":
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=self.data_path + "/video",
                **self.kwargs("replayer"),
            )
        format_converter_anonymization = OrsiFormatConverterOp(
            self,
            name="format_converter",
            pool=pool,
            **self.kwargs("format_converter_anonymization"),
        )
        anonymization_preprocessor = OrsiSegmentationPreprocessorOp(
            self,
            name="anonymization_preprocessor",
            pool=pool,
            **self.kwargs("anonymization_preprocessor")
        )

        multi_ai_inference = InferenceOp(
            self,
            name="multiai_inference",
            pool=pool,
            model_path_map = {"anonymization":self.data_path+"/model/anonymization_model.onnx"},
            **self.kwargs("multiai_inference"),
        )

        orsi_visualizer = OrsiVisualizationOp(
            self,
            name="orsi_visualizer",
            pool=pool,
            **self.kwargs("orsi_visualizer"),
        )
        self.add_flow(source, orsi_visualizer, {("", "receivers")})
        self.add_flow(source, format_converter_anonymization)
        self.add_flow(format_converter_anonymization, anonymization_preprocessor)
        self.add_flow(anonymization_preprocessor, multi_ai_inference, {("", "receivers")})
        self.add_flow(multi_ai_inference, orsi_visualizer, {("transmitter", "receivers")})
if __name__ == "__main__":
    app = OrsiInOutBodyApp()
    config_file = os.path.join(os.path.dirname(__file__), "app_config.yaml")
    app.config(config_file)
    app.run()