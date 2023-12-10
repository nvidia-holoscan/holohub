import os
from holoscan.core import Application
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import (
    InferenceOp,
    VideoStreamReplayerOp,
)
# from holoscan.videomaster import VideoMasterSourceOp

from holoscan.resources import UnboundedAllocator
from holohub.orsi_format_converter import OrsiFormatConverterOp
from holohub.orsi_segmentation_preprocessor import OrsiSegmentationPreprocessorOp
from holohub.orsi_segmentation_postprocessor import OrsiSegmentationPostprocessorOp
from holohub.orsi_visualizer import OrsiVisualizationOp

class OrsiMultiAIARApp(Application):
    def __init__(self):
        super().__init__()
        self.name = "OrsiMultiAIAR"
        self.data_path = os.path.abspath(os.environ.get("HOLOSCAN_DATA_PATH", "../data/orsi"))


    def compose(self):

        allocator=UnboundedAllocator(self, name="allocator")
        # Built-in Holoscan operators
        source_type = self.kwargs("source")['source']
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self.data_path ,
            **self.kwargs("replayer"),
        )
        multi_ai_inference = InferenceOp(
            self,
            name="multiai_inference",
            allocator=allocator,
            model_path_map = {"tool_segmentation":self.data_path+"/model/segmentation_model.onnx",
                              "anonymization": self.data_path+"/model/anonymization_model.onnx"},
            **self.kwargs("multiai_inference"),
        )
        # Orsi operators
        segmentation_preprocessor = OrsiSegmentationPreprocessorOp(
            self,
            name="segmentation_preprocessor",
            allocator=allocator,
            **self.kwargs("segmentation_preprocessor")
        )
        anonymization_preprocessor = OrsiSegmentationPreprocessorOp(
            self,
            name="anonymization_preprocessor",
            allocator=allocator,
            **self.kwargs("anonymization_preprocessor")
        )
        format_converter = OrsiFormatConverterOp(
            self,
            name="format_converter",
            allocator=allocator,
            **self.kwargs("format_converter"),
        )
        format_converter_anonymization = OrsiFormatConverterOp(
            self,
            name="format_converter_anonymization",
            allocator=allocator,
            **self.kwargs("format_converter_anonymization"),
        )
        segmentation_postprocessor = OrsiSegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=allocator,
            **self.kwargs("segmentation_postprocessor")
        )
        
        orsi_visualizer = OrsiVisualizationOp(
            self,
            name="orsi_visualizer",
            stl_file_path=self.data_path + "/stl/multi_ai/",
            registration_params_path=self.data_path + "/registration_params/multi_ai_ar.txt",
            **self.kwargs("orsi_visualizer"),
        )
        self.add_flow(source, orsi_visualizer, {("", "receivers")})
        self.add_flow(source, format_converter)
        self.add_flow(source, format_converter_anonymization)
        self.add_flow(format_converter_anonymization, anonymization_preprocessor)
        self.add_flow(anonymization_preprocessor, multi_ai_inference, {("", "receivers")})
        self.add_flow(multi_ai_inference, orsi_visualizer, {("transmitter", "receivers")})

        self.add_flow(format_converter, segmentation_preprocessor)
        self.add_flow(segmentation_preprocessor, multi_ai_inference, {("", "receivers")})
        self.add_flow(multi_ai_inference, segmentation_postprocessor, {("transmitter", "")})
        self.add_flow(segmentation_postprocessor, orsi_visualizer, {("", "receivers")})


def main():
    set_log_level(LogLevel.WARN)
    app = OrsiMultiAIARApp()
    config_file = os.path.join(os.path.dirname(__file__), "app_config.yaml")
    app.config(config_file)
    app.run()

if __name__ == "__main__":
    main()