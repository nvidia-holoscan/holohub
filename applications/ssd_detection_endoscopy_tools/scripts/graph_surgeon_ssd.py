# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--orig_model",
    type=str,
    default=r"/workspace/SSD/epoch_64.onnx",
    required=True,
    help="ONNX model file exported from PyTorch checkpoint, to change",
)
parser.add_argument(
    "--new_model",
    type=str,
    default=r"/workspace/SSD/epoch_64_nhwc_in.onnx",
    required=True,
    help="ONNX model filename to save to after changing the input channel shape to be [1, h, w, 3]",
)
parser.add_argument(
    "--nms",
    action="store_true",
    help="If true, add the op for EfficientNMS_TRT to the new ONNX model.",
)
parser.add_argument("--width", type=int, default=300, help="Width for exporting onnx model.")
parser.add_argument("--height", type=int, default=300, help="Height for exporting onnx model.")

args = parser.parse_args()
orig_model = args.orig_model
new_model = args.new_model
height = args.height
width = args.width
nms = args.nms

graph = gs.import_onnx(onnx.load(orig_model))

# Update graph input/output names
graph.inputs[0].name += "_old"
# graph.outputs[0].name += "_old"

# Insert a transpose at the network input tensor [1, 3, width, height] and rebind it to the
# new node [1, height, width, 3] be careful which one is h and which one is w
nhwc_to_nchw_in = gs.Node("Transpose", name="transpose_input", attrs={"perm": [0, 3, 1, 2]})
nhwc_to_nchw_in.outputs = graph.inputs
graph.inputs = [gs.Variable("INPUT__0", dtype=graph.inputs[0].dtype, shape=[1, height, width, 3])]
nhwc_to_nchw_in.inputs = graph.inputs
print("Changing the ONNX model input to have shape: ", 1, height, width, 3)

graph.nodes.extend([nhwc_to_nchw_in])

if nms:
    # Add NMS post-processing. Append the EfficientNMS_TRT plug-in to the network.
    # This plug-in performs non-max suppression (NMS) of the network output, which is a common
    # post-processing step for detection models.  Further info about the plug-in and its parameters
    # are available at: https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin
    max_output_boxes = 20
    attrs = OrderedDict(
        plugin_version="1",
        background_class=0,
        max_output_boxes=max_output_boxes,
        score_threshold=0.05,
        iou_threshold=0.15,
        score_activation=False,
        box_coding=0,
    )

    op_outputs = [
        gs.Variable(
            name="num_detections",
            dtype=np.int32,
            shape=["batch_size", 1],
        ),
        gs.Variable(
            name="detection_boxes",
            dtype=np.float32,
            shape=["batch_size", max_output_boxes, 4],
        ),
        gs.Variable(
            name="detection_scores",
            dtype=np.float32,
            shape=["batch_size", max_output_boxes],
        ),
        gs.Variable(
            name="detection_classes",
            dtype=np.int32,
            shape=["batch_size", max_output_boxes],
        ),
    ]

    # Create the NMS Plugin node with the selected inputs. The outputs of the node will also
    # become the final outputs of the graph.
    graph.layer(
        op="EfficientNMS_TRT",
        name="batched_nms",
        inputs=graph.outputs,
        outputs=op_outputs,
        attrs=attrs,
    )

    graph.outputs = op_outputs


graph.toposort().cleanup()

onnx.save(gs.export_onnx(graph), new_model)
