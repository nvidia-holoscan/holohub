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

import queue
from typing import Callable

import holoscan
import numpy as np
import omni
from isaacsim import SimulationApp


class Simulator:
    """A simulator environment for Isaac Sim that manages a 3D scene with dynamic objects and camera.

    This class sets up a simulator environment with:
    - A camera for capturing images
    - A dynamic texture system for real-time image updates
    - Physics simulation capabilities

    Args:
        headless (bool): Whether to run the simulation in headless mode (without GUI)
        image_size (tuple[int, int, int]): The dimensions of the camera image (width, height, channels)
        fps (float): The target frames per second for the simulation (if None, the simulation will run at the maximum possible frame rate)
        frame_count (int): The number of frames to run the simulation (-1 for infinite)
    """

    def __init__(
        self,
        headless: bool,
        image_size: tuple[int, int, int],
        fps: float,
        frame_count: int,
    ):
        self._headless = headless
        self._image_size = image_size
        self._fps = fps
        self._frame_count = frame_count

        if not (self._image_size[2] == 3 or self._image_size[2] == 4):
            raise ValueError(f"Invalid image components count: {self._image_size[2]}")

        # queue to store the data from the Holoscan application
        self._data_queue = queue.Queue(maxsize=1)

        self._simulation_app = SimulationApp({"headless": self._headless})

        # Any Omniverse level imports must occur after the SimulationApp class is instantiated
        import isaacsim.core.utils.numpy.rotations as rot_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
        from isaacsim.sensors.camera import Camera
        from isaacsim.storage.native import get_assets_root_path

        # preparing the scene
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError("Could not find Isaac Sim assets folder")

        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        # Add Franka
        asset_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
        # add robot to stage
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Arm")
        # create an articulation object
        self._arm = Articulation(prim_paths_expr="/World/Arm", name="my_arm")
        # set the initial poses of the arm so it doesn't collide BEFORE the simulation starts
        self._arm.set_world_poses(positions=np.array([[0.0, 1.0, 0.0]]) / get_stage_units())

        # create a dynamic texture for the camera image
        self._dynamic_texture = self._create_dynamic_texture(
            scope_name="holoscan", texture_name="dyn_texture"
        )

        # create a camera
        self._camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 10.0]),
            frequency=self._fps,
            resolution=(self._image_size[1], self._image_size[0]),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
            annotator_device="cuda",
        )

        # Resetting the world needs to be called before querying anything related to an articulation specifically.
        # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
        self._world.reset()

        self._camera.initialize()

    def __del__(self):
        self._simulation_app.close()

    def data_ready_callback(self, data_dict: dict):
        self._data_queue.put(data_dict)

    def _create_dynamic_texture(self, scope_name: str, texture_name: str):
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

        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()

        scope_path: str = f"{stage.GetDefaultPrim().GetPath()}/{scope_name}"

        # Create a dynamic texture provider
        dynamic_texture = omni.ui.DynamicTextureProvider(texture_name)

        # Create a material for the dynamic texture
        material_path = f"{scope_path}/{texture_name}"
        material: UsdShade.Material = UsdShade.Material.Define(stage, material_path)
        shader: UsdShade.Shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        shader.CreateIdAttr("OmniPBR")
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            f"dynamic://{texture_name}"
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Create a plane for the dynamic texture
        plane_path = "/World/plane"
        plane_geom = UsdGeom.Plane.Define(stage, plane_path)
        # Set the size, position and rotation of the plane
        plane_geom.AddTranslateOp().Set(Gf.Vec3f(2.5, 0.5, 1.5))
        plane_geom.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 26.0))
        plane_geom.AddScaleOp().Set(Gf.Vec3f(self._image_size[1] / self._image_size[0], 1, 2.0))

        # Add texture coordinates to the plane
        plane = stage.GetPrimAtPath(plane_path)
        tex_coords = UsdGeom.PrimvarsAPI(plane).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
        )
        tex_coords.Set([(0, 1), (1, 1), (1, 0), (0, 0)])
        # Bind the material to the plane
        binding_api = UsdShade.MaterialBindingAPI.Apply(plane)
        binding_api.Bind(material)

        return dynamic_texture

    def _update_dynamic_texture(self, tensor: holoscan.core.Tensor):
        """Update the dynamic texture with new image data.

        Args:
            tensor (holoscan.core.Tensor): The image data to be displayed on the dynamic texture
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
        - Pulls the data from the Holoscan application
        - Updates the dynamic texture with the image from the Holoscan application
        - Updates the arm joint positions with the joint positions from the Holoscan application

        Args:
            push_data_callback (Callable): A callback function that receives data from the simulation
        """

        current_frame = 1
        while self._simulation_app.is_running():
            if self._frame_count != -1 and current_frame > self._frame_count:
                break

            self._world.step(render=True)

            # get the data from the simulation and push it to the Holoscan application
            push_data = dict()
            if self._image_size[2] == 3:
                push_data["camera_image"] = self._camera.get_current_frame()["rgba"][:, :, :3]
            else:
                push_data["camera_image"] = self._camera.get_current_frame()["rgba"]
            push_data["arm_joint_positions"] = self._arm.get_joint_positions()
            push_data_callback(push_data)

            # get the data from the Holoscan application
            pull_data = self._data_queue.get()
            # update the dynamic texture with the image from the Holoscan application
            self._update_dynamic_texture(pull_data["camera_image"])
            # update the arm joint positions with the joint positions from the Holoscan application if they are available
            if "arm_joint_positions" in pull_data:
                # update the arm joint positions with the joint positions from the Holoscan application
                self._arm.set_joint_position_targets(pull_data["arm_joint_positions"])

            current_frame += 1
