# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

import onnx
import onnx_graphsurgeon as gs

graph = gs.import_onnx(onnx.load(sys.argv[1]))

# Update graph input/output names
graph.inputs[0].name += "_old"

# Insert a transpose at the network input tensor and rebind it to the new node (1 x 3 x 512 x 512)
nhwc_to_nchw_in = gs.Node("Transpose", name="transpose_input", attrs={"perm": [0, 3, 1, 2]})
nhwc_to_nchw_in.outputs = graph.inputs
graph.inputs = [gs.Variable("INPUT__0", dtype=graph.inputs[0].dtype, shape=[1, 640, 640, 3])]
nhwc_to_nchw_in.inputs = graph.inputs

graph.nodes.extend([nhwc_to_nchw_in])
graph.toposort().cleanup()

onnx.save(gs.export_onnx(graph), sys.argv[2])
