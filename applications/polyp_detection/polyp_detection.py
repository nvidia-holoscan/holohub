# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from argparse import ArgumentParser

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType, UnboundedAllocator

import time 
import contextlib
import collections
from collections import OrderedDict

import numpy as np
import cupy as cp
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as T 

import tensorrt as trt
import os
from torch.utils.data import Dataset, DataLoader
import json
from timeit import default_timer as timer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        
        if self.backend == 'cuda':
            self.stream = cuda.Stream()

        self.time_profile = TimeProfiler()

    def init(self, ):
        self.dynamic = False 

    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        '''build binddings
        '''
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        # max_batch_size = 1

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True 
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
                    context.set_input_shape(name, shape)

            if self.backend == 'cuda':
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 

            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)
            
            # TODO (lyuwenyu): check dtype, 
            assert self.bindings[n].data.dtype == blob[n].dtype, '{} dtype mismatch'.format(n)
            # if self.bindings[n].data.dtype != blob[n].shape:
            #     blob[n] = blob[n].to(self.bindings[n].data.dtype)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs


    def async_run_cuda(self, blob):
        '''numpy input
        '''
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)
        
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        
        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            outputs[n] = self.bindings[n].data
        
        self.stream.synchronize()
        
        return outputs
    
    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)

        elif self.backend == 'cuda':
            return self.async_run_cuda(blob)

    def synchronize(self, ):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

        elif self.backend == 'cuda':
            self.stream.synchronize()
    
    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)

    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)

        return self.time_profile.total / n 


    @staticmethod
    def onnx2tensorrt():
        pass

class CustomInferenceOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig_size = (1164, 1034)
        self.device = device
        self.model = TRTInference("/colon_workspace/rt_detrv2_timm_r50_nvimagenet_pretrained_demo.trt", device=device)

    def setup(self, spec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        # Get the output from the FormatConverterOp
        preprocessed_frame = op_input.receive("input")
        
        images = torch.tensor(preprocessed_frame['source_video']).to(device)
        if images.ndim == 3:
            # h, w, c -> n, c, h, w
            images = images.permute(2, 0, 1).unsqueeze(0)
        # Perform inference using the TensorRT model
        input_data = {
            'images': images, 
            'orig_target_sizes': torch.tensor(self.orig_size)[None].to(device),
        }

        output = self.model(input_data)

        op_output.emit(output, "output")

class DetectionPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        # spec.output("out")

    def compute(self, op_input, op_output, context, scores_threshold=0.5):
        # Get input message which is a dictionary
        in_message = op_input.receive("in")
        bboxes = cp.asarray(in_message["boxes"])
        scores = cp.asarray(in_message["scores"])
        # bboxes: (batch, nboxes, 4), scores: (batch, nboxes)
        filtered_data = []
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            mask = score >= scores_threshold
            if cp.any(mask):
                filtered_bboxes = bbox[mask].tolist()
            else:
                filtered_bboxes = []
            filtered_data.append(filtered_bboxes)
            print(f"detect {len(filtered_bboxes)} objects")

class PolypDetectionApp(Application):
    def __init__(self, data, source="replayer"):
        """Initialize the colonoscopy detection application

        Parameters
        ----------
        source : {"replayer"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        """

        super().__init__()

        # set name
        self.name = "Polyp Detection App"

        # Optional parameters affecting the graph created by compose.
        self.source = source
        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        n_channels = 3
        bpp = 4  # bytes per pixel

        video_dir = os.path.join(self.sample_data_path)
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self, name="replayer", directory=video_dir, **self.kwargs("replayer")
        )

        width_preprocessor = 640
        height_preprocessor = 640
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 2
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            **self.kwargs("detection_preprocessor"),
        )

        detection_inference = CustomInferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="pool"),
            **self.kwargs("detection_inference"),
        )

        detection_postprocessor = DetectionPostprocessorOp(
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        # detection_visualizer = HolovizOp(
        #     self,
        #     name="detection_visualizer",
        #     tensors=[
        #         dict(name="", type="color"),
        #         dict(
        #             name="rectangles",
        #             type="rectangles",
        #             opacity=0.5,
        #             line_width=4,
        #             color=[1.0, 0.0, 0.0, 1.0],
        #         ),
        #     ],
        #     **self.kwargs("detection_visualizer"),
        # )

        # self.add_flow(source, detection_visualizer, {("", "receivers")})
        self.add_flow(source, detection_preprocessor)
        self.add_flow(detection_preprocessor, detection_inference, {("tensor", "input")})
        self.add_flow(detection_inference, detection_postprocessor, {("output", "in")})
        # self.add_flow(detection_postprocessor, detection_visualizer, {("out", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Colonoscopy segmentation demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="/colon_workspace/holohub/data/polyp_detection",
        help=("Set the data path"),
    )

    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "polyp_detection.yaml")
    else:
        config_file = args.config
    app = PolypDetectionApp(data=args.data, source=args.source)
    app.config(config_file)
    app.run()
