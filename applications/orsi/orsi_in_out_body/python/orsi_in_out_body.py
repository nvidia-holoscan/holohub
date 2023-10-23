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
        if source_type == "replayer":
            source = VideoStreamReplayerOp(
                self,
                name="replayer",
                directory=self.data_path + "/video",
                **self.kwargs("replayer"),
            )
        

if __name__ == "__main__":
    app = OrsiInOutBodyApp()
    config_file = os.path.join(os.path.dirname(__file__), "app_config.yaml")
    app.config(config_file)
    app.run()