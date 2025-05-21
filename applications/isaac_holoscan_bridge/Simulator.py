# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import omni
import logging
import numpy as np

import holoscan

from isaacsim import SimulationApp
from typing import Callable


class Simulator:
    """A simulator environment for IsaacSim that manages a 3D scene with dynamic objects and camera.

    This class sets up a simulator environment with:
    - A dynamic scene containing cubes with different properties
    - A camera for capturing images
    - A dynamic texture system for real-time image updates
    - Physics simulation capabilities

    Args:
        headless (bool): Whether to run the simulation in headless mode (without GUI)
        image_size (tuple[int, int, int]): The dimensions of the camera image (width, height, channels)
    """

    def __init__(
        self,
        headless: bool,
        image_size: tuple[int, int, int],
        fps: float,
    ):
        self._headless = headless
        self._image_size = image_size
        self._fps = fps

        self._simulation_app = SimulationApp({"headless": self._headless})

        # Any Omniverse level imports must occur after the SimulationApp class is instantiated
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.api.objects import DynamicCuboid
        import isaacsim.core.utils.numpy.rotations as rot_utils
        from pxr import UsdShade, Sdf, UsdGeom

        self._world = World(stage_units_in_meters=1.0)

        def create_dynamic_texture(scope_name: str, texture_name: str):
            """Creates a dynamic texture in the USD stage that can be updated in real-time.

            This function sets up a dynamic texture system that allows for real-time updates of image data.
            It creates a material with a dynamic texture input and applies it to a plane in the scene.
            The texture can be updated using the returned DynamicTextureProvider.

            Args:
                scope_name (str): The name of the scope where the texture will be created in the USD stage
                texture_name (str): The name of the texture to be created

            Returns:
                omni.ui.DynamicTextureProvider: A provider object that can be used to update the texture data
            """
            stage = omni.usd.get_context().get_stage()

            scope_path: str = f"{stage.GetDefaultPrim().GetPath()}/{scope_name}"

            magenta = np.array([255, 0, 255, 255], np.uint8)
            frame = np.full((1, 1, 4), magenta, dtype=np.uint8)
            height, width, channels = frame.shape
            dynamic_texture = omni.ui.DynamicTextureProvider(texture_name)
            dynamic_texture.set_data_array(frame, [width, height, channels])

            material_path = f"{scope_path}/{texture_name}"
            material: UsdShade.Material = UsdShade.Material.Define(stage, material_path)
            shader: UsdShade.Shader = UsdShade.Shader.Define(
                stage, f"{material_path}/Shader"
            )
            shader.SetSourceAsset("OmniPBR.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            shader.CreateIdAttr("OmniPBR")
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                f"dynamic://{texture_name}"
            )
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )

            plane_path = "/World/plane"
            UsdGeom.Plane.Define(stage, plane_path)
            plane = stage.GetPrimAtPath(plane_path)
            texCoords = UsdGeom.PrimvarsAPI(plane).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
            )
            texCoords.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
            binding_api = UsdShade.MaterialBindingAPI.Apply(plane)
            binding_api.Bind(material)

            return dynamic_texture

        # create a dynamic texture for the camera image
        self._dynamic_texture = create_dynamic_texture(
            scope_name="holoscan", texture_name="dyn_texture"
        )

        self._camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 10.0]),
            frequency=self._fps,
            resolution=(self._image_size[1], self._image_size[0]),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 90, 0]), degrees=True
            ),
            annotator_device="cuda",
        )

        try:
            self._world.scene.add_default_ground_plane()
        except Exception as e:
            logging.error(
                f"Error adding default ground plane: {e}. Maybe running develop version and not logged in to Omniverse Nucleus server?"
            )

        # Resetting the world needs to be called before querying anything related to an articulation specifically.
        # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
        self._world.reset()

        self._camera.initialize()

    def __del__(self):
        self._simulation_app.close()

    def get_image_size(self):
        """Get the dimensions of the camera image.

        Returns:
            tuple[int, int, int]: The image dimensions (width, height, channels)
        """
        return self._image_size

    def data_ready_callback(self, tensor: holoscan.core.Tensor):
        """Update the dynamic texture with new image data.

        Args:
            data: The image data to be displayed on the dynamic texture
        """
        if len(tensor.shape) != 3:
            raise ValueError(f"Invalid tensor shape: {tensor.shape}")
        dtype = tensor.dtype
        if dtype.bits == 8:
            if tensor.shape[2] == 3:
                format = omni.ui.TextureFormat.RGB8_UNORM
            elif tensor.shape[2] == 4:
                format = omni.ui.TextureFormat.RGBA8_UNORM
            else:
                raise ValueError(f"Invalid number of channels: {tensor.shape[2]}")
        elif dtype.bits == 16:
            if tensor.shape[2] == 3:
                format = omni.ui.TextureFormat.RGB16_UNORM
            elif tensor.shape[2] == 4:
                format = omni.ui.TextureFormat.RGBA16_UNORM
            else:
                raise ValueError(f"Invalid number of channels: {tensor.shape[2]}")
        else:
            raise ValueError(f"Unsupported tensor dtype: {dtype}")

        self._dynamic_texture.set_bytes_data_from_gpu(
            gpu_bytes=tensor.data,
            sizes=[tensor.shape[1], tensor.shape[0]],
            format=format,
            stride=tensor.strides[0],
        )

    def run(self, push_data_callback: Callable):
        """Run the simulation loop.

        This method starts the simulation and continuously:
        - Steps the physics simulation
        - Renders the scene
        - Pushes the data to the Holoscan application

        Args:
            push_data_callback (Callable): A callback function that receives data from the simulation

        Raises:
            ValueError: If the image size has an invalid number of channels (must be 3 for RGB or 4 for RGBA)
        """
        if not (self._image_size[2] == 3 or self._image_size[2] == 4):
            raise ValueError(f"Invalid image components count: {self._image_size[2]}")

        while self._simulation_app.is_running():
            self._world.step(render=True)

            # get the data an push it to the Holoscan application
            data = dict()
            if self._image_size[2] == 3:
                data["camera_image"] = self._camera.get_current_frame()["rgba"][
                    :, :, :3
                ]
            else:
                data["camera_image"] = self._camera.get_current_frame()["rgba"]
            data["camera_pose"] = self._camera.get_world_pose()
            push_data_callback(data)
