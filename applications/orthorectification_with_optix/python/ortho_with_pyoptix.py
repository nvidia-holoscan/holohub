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


import os
import time

import cupy as cp
import holoscan as hs
import numpy as np
import tqdm
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import HolovizOp
from odm_utils import load_all_frames_odm
from optix_utils_for_ortho import (
    State,
    build_sensor_geom,
    build_terrain_geom,
    compile_cuda,
    create_accel,
    create_ctx,
    create_module,
    create_pipeline,
    create_program_groups,
    create_sbt,
    init_render_params,
    launch,
    load_sensor_texture,
    load_terrain_host,
    set_pipeline_options,
)
from ortho_utils import create_blank_basemap, extract_extent_nativeCRS, load_basemap
from osgeo import gdal
from shapely.wkt import loads as wkt_load

# dataset paths within the container
data_dir = os.path.expanduser("~/Data/lafayette_square")
image_dir = os.path.join(data_dir, "images")
geo_path = os.path.join(data_dir, "odm_georeferencing", "odm_georeferencing_model_geo.txt")
dem_image_fname = os.path.join(data_dir, "odm_dem", "dsm_small_filled.tif")
basemap_fname = None

# config and include paths within container
cuda_tk_path = "/usr/local/cuda-11.6/include"
include_path = os.path.normpath("/work/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64/include")
proot = os.path.normpath("/work/")
project_include_path = os.path.normpath("/work/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64/SDK")
project_cu = os.path.join(proot, "cpp", "src", "optix", "optixOrtho.cu")

# general configuration parameters
sensor_dims = (5472, 3648)
sensor_resize = 0.25  # resizes the raw sensor pixels
ncpu = 8  # how many cores to use to load sensor simulation
gsd = 0.25  # controls how many pixels are in the rendering
iterations = 18  # how many frames to render from the source images (in this case 425 is max)
use_mosaic_bbox = True  # render to a static bounds on the ground as defined by the DEM
write_geotiff = False
nb = 3  # how many bands to write to the GeoTiff
render_scale = 0.5  # scale the holoview window up or down
fps = 8.0  # rate limit the simulated sensor feed to this many frames per second

# ---------------Helper Functionality -------------------------#
# TODO Extend to camera models other than perspective

if sensor_resize is not None:
    sensor_dims = [
        int(item) for item in (sensor_dims[0] * sensor_resize, sensor_dims[1] * sensor_resize)
    ]

pbar = tqdm.tqdm(total=iterations, desc="ortho frames")

frame_rate = 1 / fps
dem_bounds = extract_extent_nativeCRS(dem_image_fname)
xmin, ymin, xmax, ymax = dem_bounds

box_size_x = xmax - xmin
box_size_y = ymax - ymin

render_width = int(box_size_x / gsd)
render_height = int(box_size_y / gsd)


if use_mosaic_bbox:
    print("Creating mosaic basemap")
    if basemap_fname is not None:
        basemap = load_basemap(basemap_fname, render_width, render_height)
    else:
        basemap = create_blank_basemap(render_width, render_height)

    mosaic_d = cp.array(basemap)
    mosaic_alpha_d = cp.full((render_height, render_width, 1), 255, cp.uint8)

    mosaic_image_d = cp.dstack([mosaic_d, mosaic_alpha_d])


# ---------------BEGIN HOLOSCAN -------------------------#
class FrameSimulatorODMOp(Operator):
    def __init__(self, *args, **kwargs):
        self.frame_count = 0
        self.prior_time = time.time()
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        print("pre-fetch frames from disk to simulate streaming sensor")

        self.staged_dataset = load_all_frames_odm(
            use_mosaic_bbox,
            dem_image_fname,
            geo_path,
            data_dir,
            iterations,
            ncpu,
            sensor_resize,
            sensor_dims,
        )

        spec.output("sensor_meta")
        spec.output("sensor_pix")

    def compute(self, op_input, op_output, context):
        data = self.staged_dataset[self.frame_count]

        sensor_pix = data["sensor_pix"]
        meta = data["meta"]

        self.frame_count = self.frame_count + 1
        pbar.update(1)

        curr_time = time.time()
        delta_time = curr_time - self.prior_time
        if delta_time < frame_rate:
            time.sleep(frame_rate - delta_time)
        self.prior_time = time.time()

        op_output.emit(meta, "sensor_meta")
        op_output.emit(sensor_pix, "sensor_pix")


class LoadTerrain(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        self.state = State()
        self.state.image_gsd = np.float64(gsd)
        self.state.sensor_alpha = cp.full((sensor_dims[1], sensor_dims[0], 1), 255, cp.uint8)
        if use_mosaic_bbox:
            self.state.image = mosaic_image_d

        load_terrain_host(self.state, dem_image_fname)
        build_terrain_geom(self.state)

        spec.input("sensor_meta")
        spec.input("sensor_pix")

        spec.output("optix_state")
        spec.output("sensor_meta")
        spec.output("sensor_pix")

    def compute(self, op_input, op_output, context):
        sensor_meta = op_input.receive("sensor_meta")
        sensor_pix = op_input.receive("sensor_pix")

        # TODO dynamically update / load terrain based on image footprint location

        op_output.emit(self.state, "optix_state")
        op_output.emit(sensor_meta, "sensor_meta")
        op_output.emit(sensor_pix, "sensor_pix")


class PrepOdmMeta4Ortho(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        self.state = State()

        spec.input("sensor_meta")
        spec.input("sensor_pix")
        spec.input("optix_state")

        spec.output("optix_state")
        spec.output("sensor_pix")

    def compute(self, op_input, op_output, context):
        sensor_meta = op_input.receive("sensor_meta")
        sensor_pix = op_input.receive("sensor_pix")
        optix_state = op_input.receive("optix_state")

        optix_state.frame_number = sensor_meta["frame_number"]
        optix_state.bounding_box_estimate = wkt_load(sensor_meta["wkt"]).bounds
        optix_state.sensor_dir = np.array(
            [float(item) for item in sensor_meta["sensor_dir"].split(",")]
        )
        optix_state.sensor_up = np.array(
            [float(item) for item in sensor_meta["sensor_up"].split(",")]
        )

        optix_state.sensor_tex_width = sensor_dims[0]
        optix_state.sensor_tex_height = sensor_dims[1]

        optix_state.sensor_hor_fov = np.deg2rad(float(sensor_meta["hfov"]))
        optix_state.sensor_ver_fov = np.deg2rad(float(sensor_meta["vfov"]))

        optix_state.utm_sensor_pos = np.array(
            [float(item) for item in sensor_meta["utm_pos"].split(",")]
        )

        init_render_params(optix_state)

        if not use_mosaic_bbox:
            optix_state.image = cp.zeros(
                (optix_state.image_width, optix_state.image_height, 4), cp.uint8
            )

        op_output.emit(sensor_pix, "sensor_pix")
        op_output.emit(optix_state, "optix_state")


class OptixOrthoOp(Operator):
    def __init__(self, *args, **kwargs):
        self.project_ptx = compile_cuda(
            project_cu, cuda_tk_path, include_path, project_include_path
        )
        self.ctx = create_ctx()
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        pipeline_options = set_pipeline_options()
        module = create_module(self.ctx, pipeline_options, self.project_ptx)
        self.prog_groups = create_program_groups(self.ctx, module)

        self.pipeline = create_pipeline(self.ctx, self.prog_groups, pipeline_options)
        self.stream = cp.cuda.Stream()

        spec.input("optix_state")
        spec.input("sensor_pix")

        spec.output("gtiff_meta")
        spec.output("ortho_pix")

    def compute(self, op_input, op_output, context):
        optix_state = op_input.receive("optix_state")
        sensor_pix = op_input.receive("sensor_pix")

        load_sensor_texture(optix_state, sensor_pix)
        build_sensor_geom(optix_state)

        gas_handle, d_gas_output_buffer = create_accel(optix_state, self.ctx, self.stream)

        sbt = create_sbt(self.prog_groups)
        ortho_pix = launch(optix_state, self.pipeline, sbt, gas_handle, self.stream)

        gtiff_meta = {}
        gtiff_meta["xsize"] = optix_state.image_width
        gtiff_meta["ysize"] = optix_state.image_height
        gtiff_meta["geot"] = optix_state.image_geot
        gtiff_meta["wkt"] = optix_state.image_wkt
        gtiff_meta["frame_number"] = optix_state.frame_number

        op_output.emit(gtiff_meta, "gtiff_meta")
        op_output.emit(ortho_pix, "ortho_pix")


class ProcessOrthoOp(Operator):
    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("gtiff_meta")
        spec.input("ortho_pix")

        spec.output("outputs")

    def compute(self, op_input, op_output, context):
        ortho_meta = op_input.receive("gtiff_meta")
        ortho_pix = op_input.receive("ortho_pix")

        if write_geotiff:
            img = ortho_pix[:, :, 0:nb]

            frame_str = str(ortho_meta["frame_number"]).zfill(8)
            ortho_fname = os.path.join(data_dir, frame_str + "_frame_pyoptix_ortho.tif")

            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(
                ortho_fname,
                xsize=ortho_meta["xsize"],
                ysize=ortho_meta["ysize"],
                bands=nb,
                eType=gdal.GDT_Byte,
                options=["TILED=YES", "COMPRESS=LZW"],
            )

            dst_ds.SetGeoTransform(ortho_meta["geot"])
            dst_ds.SetProjection(ortho_meta["wkt"])
            for bn in range(nb):
                band = dst_ds.GetRasterBand(bn + 1)
                band.SetNoDataValue(0)

                band.WriteArray(np.squeeze(img[:, :, bn]))

            dst_ds = None

        out_message = Entity(context)
        out_message.add(hs.as_tensor(ortho_pix), "pixels")

        op_output.emit(out_message, "outputs")

    def on_error(e):
        print("There was an error: {}".format(e))


# ---------------END HOLOSCAN-------------------------#
class BasicOrthoFlow(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        src = FrameSimulatorODMOp(self, CountCondition(self, iterations), name="src")
        meta = PrepOdmMeta4Ortho(self, name="prep_odm_meta")

        terrain = LoadTerrain(self, name="terrain_loader")
        ortho = OptixOrthoOp(self, name="optix_ortho")
        ortho_proc = ProcessOrthoOp(self, name="proc_ortho")

        rh = int(render_height * render_scale)
        rw = int(render_width * render_scale)
        sink = HolovizOp(
            self, name="holoviz ortho viewer", width=rw, height=rh, **self.kwargs("holoviz")
        )

        self.add_flow(src, terrain, {("sensor_meta", "sensor_meta"), ("sensor_pix", "sensor_pix")})
        self.add_flow(
            terrain,
            meta,
            {
                ("sensor_meta", "sensor_meta"),
                ("sensor_pix", "sensor_pix"),
                ("optix_state", "optix_state"),
            },
        )
        self.add_flow(meta, ortho, {("optix_state", "optix_state"), ("sensor_pix", "sensor_pix")})
        self.add_flow(ortho, ortho_proc, {("gtiff_meta", "gtiff_meta"), ("ortho_pix", "ortho_pix")})
        self.add_flow(ortho_proc, sink, {("outputs", "receivers")})


if __name__ == "__main__":
    app = BasicOrthoFlow()
    app.config("")
    app.run()
    pbar.close()

    print("done with app")
