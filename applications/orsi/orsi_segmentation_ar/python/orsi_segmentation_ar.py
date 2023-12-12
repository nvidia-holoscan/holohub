import os

from holoscan.core import Application
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import InferenceOp, VideoStreamReplayerOp
from holoscan.resources import UnboundedAllocator

from holohub.orsi_format_converter import OrsiFormatConverterOp
from holohub.orsi_segmentation_postprocessor import OrsiSegmentationPostprocessorOp
from holohub.orsi_segmentation_preprocessor import OrsiSegmentationPreprocessorOp
from holohub.orsi_visualizer import OrsiVisualizationOp

# from holoscan.videomaster import VideoMasterSourceOp


class OrsiSegmentationARApp(Application):
    def __init__(self):
        super().__init__()
        self.name = "OrsiSegmentationAR"
        self.data_path = os.path.abspath(os.environ.get("HOLOSCAN_DATA_PATH", "../data/orsi"))

    def compose(self):
        allocator = UnboundedAllocator(self, name="allocator")
        # Built-in Holoscan operators
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self.data_path,
            **self.kwargs("replayer"),
        )
        multi_ai_inference = InferenceOp(
            self,
            name="multiai_inference",
            allocator=allocator,
            model_path_map={"tool_segmentation": self.data_path + "/model/segmentation_model.onnx"},
            **self.kwargs("multiai_inference"),
        )
        # Orsi operators
        segmentation_preprocessor = OrsiSegmentationPreprocessorOp(
            self,
            name="segmentation_preprocessor",
            allocator=allocator,
            **self.kwargs("segmentation_preprocessor"),
        )
        format_converter = OrsiFormatConverterOp(
            self,
            name="format_converter",
            allocator=allocator,
            **self.kwargs("format_converter"),
        )
        segmentation_postprocessor = OrsiSegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=allocator,
            **self.kwargs("segmentation_postprocessor"),
        )
        orsi_visualizer = OrsiVisualizationOp(
            self,
            name="orsi_visualizer",
            stl_file_path=self.data_path + "/stl/segmentation/",
            registration_params_path=self.data_path + "/registration_params/segmentation_ar.txt",
            **self.kwargs("orsi_visualizer"),
        )
        self.add_flow(source, orsi_visualizer, {("", "receivers")})
        self.add_flow(source, format_converter)
        self.add_flow(format_converter, segmentation_preprocessor)
        self.add_flow(segmentation_preprocessor, multi_ai_inference, {("", "receivers")})
        self.add_flow(multi_ai_inference, segmentation_postprocessor, {("transmitter", "")})
        self.add_flow(segmentation_postprocessor, orsi_visualizer, {("", "receivers")})


if __name__ == "__main__":
    set_log_level(LogLevel.WARN)
    app = OrsiSegmentationARApp()
    config_file = os.path.join(os.path.dirname(__file__), "app_config.yaml")
    app.config(config_file)
    app.run()
