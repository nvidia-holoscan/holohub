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
        self.data_path = os.environ.get("HOLOSCAN_DATA_PATH", "../data/orsi")


    def compose(self):

        allocator=UnboundedAllocator(self, name="allocator")
        # Built-in Holoscan operators
        source_type = self.kwargs("source")['source']
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self.data_path + "/video",
            **self.kwargs("replayer"),
        )
        multi_ai_inference = InferenceOp(
            self,
            name="multiai_inference",
            allocator=allocator,
            model_path_map = {"anonymization":self.data_path+"/model/anonymization_model.onnx"},
            **self.kwargs("multiai_inference"),
        )
        # Orsi operators
        anonymization_preprocessor = OrsiSegmentationPreprocessorOp(
            self,
            name="anonymization_preprocessor",
            allocator=allocator,
            **self.kwargs("anonymization_preprocessor")
        )
        format_converter_anonymization = OrsiFormatConverterOp(
            self,
            name="format_converter",
            allocator=allocator,
            **self.kwargs("format_converter_anonymization"),
        )
        orsi_visualizer = OrsiVisualizationOp(
            self,
            name="orsi_visualizer",
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