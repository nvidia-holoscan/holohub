# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import logging
import sys
from pathlib import Path

import numpy as np
from holoscan.core import ConditionType, Fragment, Operator, OperatorSpec
from pxr import Kind, Tf, Usd, UsdGeom
from stl import mesh

logger = logging.getLogger("HoloscanMeshToUSD")


class SendMeshToUSDOp(Operator):
    """
    Converts a 3D mesh in STL format from a bytestream or file
    to UsdGeom mesh format in an existing OpenUSD scene.
    """

    ELEMENTS_PER_POINT = 3
    POINTS_PER_CELL = 3

    def __init__(
        self,
        fragment: Fragment,
        *args,
        g_stage=None,
        stl_file_path=None,
        **kwargs,
    ):
        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        self.mesh_path = stl_file_path
        self.input_name_stl_bytes = "stl_bytes"  # Alternatively can use a class attribute
        self.existing_stage = g_stage
        # Call the base class __init__() last.
        # Also, the base class has an attribute called fragment for storing the fragment object
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        # The current STL operator has a named output for STL bytes, so this operator will use it.
        # For ease of testing of this operator standalone, the file path to a STL file will be used
        # hence making the named input Optional
        spec.input(self.input_name_stl_bytes).condition(ConditionType.NONE)  # Optional input.

    def stop(self):
        # save stage before shutting down the client
        # Save the proper edit target (in the case that we're live editing)
        edit_target_layer = self.existing_stage.GetEditTarget().GetLayer()
        edit_target_layer.Save()

    def compute(self, op_input, stageURL, size):
        # If STL file path is given, use it, otherwise, get the bytes from the input
        # as IOStream to read from
        if self.mesh_path:
            if not Path.is_file(self.mesh_path):
                raise ValueError("STL file path is not valid.")
            md_mesh = mesh.Mesh.from_file(self.mesh_path)
        else:
            mesh_bytes = op_input.receive(self.input_name_stl_bytes)
            if not mesh_bytes:
                raise ValueError("Expected input, STL bytes, not received.")
            with io.BytesIO(mesh_bytes) as inmemf:
                md_mesh = mesh.Mesh.from_file(None, fh=inmemf)

        default_prim_path = self.existing_stage.GetDefaultPrim().GetPath().pathString
        meshName = "mesh"
        meshPrimPath = default_prim_path + "/" + Tf.MakeValidIdentifier(meshName)
        ov_mesh = UsdGeom.Mesh.Define(self.existing_stage, meshPrimPath)
        if not ov_mesh:
            self.logger.error("[ERROR] Failure to create USD mesh based on STL mesh")
            sys.exit(1)

        self.logger.info(f"Mesh normal count: {len(md_mesh.normals)}")
        self.logger.info(f"Mesh points count: {len(md_mesh.points * self.POINTS_PER_CELL)}")
        points = md_mesh.points

        ov_mesh.CreatePointsAttr(md_mesh.points)
        ov_mesh.CreateNormalsAttr(md_mesh.normals)

        ov_mesh.CreateExtentAttr(ov_mesh.ComputeExtent(md_mesh.points))

        # create a new index for each point
        ov_mesh.CreateFaceVertexIndicesAttr(np.arange(len(points) * self.POINTS_PER_CELL))

        # Add face vertex count
        faceVertexCounts = np.full(len(points), 3)
        self.logger.info(f"Face vertex count: {faceVertexCounts}")
        ov_mesh.CreateFaceVertexCountsAttr(faceVertexCounts)

        # Make the kind a component to support the assembly/component selection hierarchy
        Usd.ModelAPI(ov_mesh.GetPrim()).SetKind(Kind.Tokens.component)
