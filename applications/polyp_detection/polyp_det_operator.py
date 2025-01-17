import collections
import contextlib
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict

import cupy as cp
import numpy as np
import tensorrt as trt
import torch
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolypDetInferenceOp(Operator):
    def __init__(self, *args, orig_size=None, trt_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig_size = orig_size
        self.device = device
        self.model = TRTInference(
            trt_model,
            device=device,
            max_batch_size=1,
        )

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        preprocessed_frame = op_input.receive("input")

        images = torch.tensor(preprocessed_frame["source_video"]).to(device)
        if images.ndim == 3:
            # h, w, c -> n, c, h, w
            images = images.permute(2, 0, 1).unsqueeze(0)
        # Perform inference using the TensorRT model
        input_data = {
            "images": images,
            "orig_target_sizes": torch.tensor(self.orig_size)[None].type(torch.int32).to(device),
        }

        output = self.model(input_data)

        op_output.emit(output, "output")


class PolypDetPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context, scores_threshold=0.5):
        # Get input message which is a dictionary
        in_message = op_input.receive("in")
        bboxes = cp.asarray(in_message["boxes"])
        scores = cp.asarray(in_message["scores"])
        # bboxes: (batch, nboxes, 4), scores: (batch, nboxes)
        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            selected_output = []
            for b, s in zip(bbox, score):
                if s > scores_threshold:
                    selected_output.append((b, s))
            if len(selected_output) > 0:
                print(
                    f"detect {len(selected_output)} boxes, highest score: {selected_output[0][1]}"
                )
            else:
                print("No boxes detected")


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(
        self,
    ):
        self.total = 0

    def __enter__(
        self,
    ):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(
        self,
    ):
        self.total = 0

    def time(
        self,
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):
    def __init__(self, engine_path, device="cuda:0", max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(
            self.engine, self.context, self.max_batch_size, self.device
        )
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        """load engine"""
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(
        self,
    ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(
        self,
    ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        """build binddings"""
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        """torch input"""
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def __call__(self, blob):
        return self.run_torch(blob)

    def synchronize(
        self,
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

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
