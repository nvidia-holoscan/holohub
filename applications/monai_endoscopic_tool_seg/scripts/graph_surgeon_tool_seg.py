# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse

import onnx
import onnx_graphsurgeon as gs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--orig_model",
    type=str,
    default=r"model_endoscopic_tool_seg_sanitized.onnx",
    required=True,
    help="ONNX model file exported from PyTorch checkpoint, to change",
)
parser.add_argument(
    "--new_model",
    type=str,
    default=r"model_endoscopic_tool_seg_sanitized_nhwc_in_nchw_out.onnx",
    required=True,
    help="ONNX model filename to save to after changing the input channel "
    "shape to be [1, h, w, 3] and output shape to nchw",
)
parser.add_argument("--width", type=int, default=736, help="Width for exporting onnx model.")
parser.add_argument("--height", type=int, default=480, help="Height for exporting onnx model.")

args = parser.parse_args()
orig_model = args.orig_model
new_model = args.new_model
height = args.height
width = args.width
out_channels = 2

graph = gs.import_onnx(onnx.load(orig_model))

# Update graph input/output names
graph.inputs[0].name += "_old"
graph.outputs[0].name += "_old"


# Insert a transpose before the network input tensor [1,3,736,480] and rebind old
# input node to the new input node [1, 480, 736, 3]
ncwh_to_nhwc_in = gs.Node("Transpose", name="transpose_input", attrs={"perm": [0, 3, 2, 1]})
ncwh_to_nhwc_in.outputs = graph.inputs
graph.inputs = [gs.Variable("INPUT__0", dtype=graph.inputs[0].dtype, shape=[1, height, width, 3])]
ncwh_to_nhwc_in.inputs = graph.inputs
print("Changing the ONNX model input to have shape: ", 1, height, width, 3)

# Insert a transpose at the network output tensor (1, 2, 736, 480) and rebind it to  (1, 2, 480, 736)
ncwh_to_nchw_out = gs.Node("Transpose", name="transpose_output", attrs={"perm": [0, 1, 3, 2]})
ncwh_to_nchw_out.inputs = graph.outputs
graph.outputs = [
    gs.Variable("OUTPUT__0", dtype=graph.outputs[0].dtype, shape=[1, out_channels, height, width])
]
ncwh_to_nchw_out.outputs = graph.outputs
print("Changing the ONNX model output to have shape: ", 1, out_channels, height, width)

graph.nodes.extend([ncwh_to_nhwc_in, ncwh_to_nchw_out])


graph.toposort().cleanup()

onnx.save(gs.export_onnx(graph), new_model)
