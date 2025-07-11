# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import ctypes  # C interop helpers
import os

import cupy as cp
import numpy as np
import optix
from osgeo import gdal
from pynvrtc.compiler import Program


class State:
    def __init__(self):
        self.bounding_box_estimate = None
        self.utm_sensor_pos = None
        self.frame_number = None
        self.image = None
        self.image_width = None
        self.image_height = None
        self.image_corner_coords = None
        self.image_gsd = None
        self.image_geot = None
        self.image_wkt = None

        self.terrain_pix = None
        self.terrain_width = None
        self.terrain_height = None
        self.terrain_gsd = None
        self.terrain_geot = None
        self.terrain_wkt = None
        self.terrain_xmin = None
        self.terrain_ymin = None
        self.terrain_zmax = None
        self.terrain_vert_grid = None
        self.terrain_tri_grid = None

        self.sensor_isRGB = None
        self.sensor_hor_fov = None
        self.sensor_ver_fov = None
        self.sensor_focal_length = None
        self.sensor_pos = None
        self.sensor_lookat = None
        self.sensor_dir = None
        self.sensor_up = None
        self.sensor_right = None
        self.sensor_focal_plane_origin = None
        self.sensor_focal_plane_size = None
        self.sensor_focal_plane_center = None
        self.half_focal_plane_x = None
        self.half_focal_plane_y = None
        self.sensor_tex_pixels = None
        self.sensor_tex_width = None
        self.sensor_tex_height = None
        self.sensor_vert_grid = None
        self.sensor_tri_grid = None

        self.sensor_alpha = None


###################
## Terrain Stuff
###################


def build_terrain_geom(state: State):
    # build verts
    x = (
        np.arange(0, state.terrain_width) * state.terrain_geot[1]
        + state.terrain_geot[0]
        - state.terrain_xmin
    )
    y = (
        np.arange(0, state.terrain_height) * state.terrain_geot[5]
        + state.terrain_geot[3]
        - state.terrain_ymin
    )
    xx, yy = np.meshgrid(x, y)
    zz = state.terrain_pix
    vert_grid = np.vstack((xx, yy, zz)).reshape([3, -1]).transpose()

    # build triangles
    ai = np.arange(0, state.terrain_width - 1)
    aj = np.arange(0, state.terrain_height - 1)
    aii, ajj = np.meshgrid(ai, aj)
    a = aii + ajj * state.terrain_width
    a = a.flatten()

    tria = np.vstack(
        (
            a,
            a + state.terrain_width,
            a + state.terrain_width + 1,
            a,
            a + state.terrain_width + 1,
            a + 1,
        )
    )
    tri_grid = np.transpose(tria).reshape([-1, 3])

    state.terrain_vert_grid = vert_grid.reshape(vert_grid.size)
    state.terrain_tri_grid = tri_grid.reshape(tri_grid.size)


def load_terrain_host(state: State, terrain_fname):
    dset = gdal.Open(terrain_fname, gdal.GA_ReadOnly)

    nx = dset.RasterXSize
    ny = dset.RasterYSize
    state.terrain_width = nx
    state.terrain_height = ny

    geot = dset.GetGeoTransform()
    state.terrain_wkt = dset.GetProjection()

    state.terrain_geot = geot
    state.terrain_xmin = geot[0] + geot[1]
    state.terrain_ymin = geot[3] + (ny - 1) * geot[5]
    state.terrain_gsd = geot[1]

    state.terrain_pix = dset.GetRasterBand(1).ReadAsArray()
    state.terrain_zmax = np.max(state.terrain_pix)

    dset = None


###################
## Sensor Stuff
###################


def build_sensor_geom(state: State):
    vert_grid = []
    tri_grid = []

    # top left corner
    for i in range(3):
        vert_grid.append(
            state.sensor_focal_plane_center[i]
            - state.half_focal_plane_x[i]
            + state.half_focal_plane_y[i]
        )

    # top right corner
    for i in range(3):
        vert_grid.append(
            state.sensor_focal_plane_center[i]
            + state.half_focal_plane_x[i]
            + state.half_focal_plane_y[i]
        )

    # bottom left corner
    for i in range(3):
        vert_grid.append(
            state.sensor_focal_plane_center[i]
            - state.half_focal_plane_x[i]
            - state.half_focal_plane_y[i]
        )

    # bottom right corner
    for i in range(3):
        vert_grid.append(
            state.sensor_focal_plane_center[i]
            + state.half_focal_plane_x[i]
            - state.half_focal_plane_y[i]
        )

    # lower left triangle
    tri_grid.extend([0, 3, 2])
    tri_grid.extend([0, 1, 3])

    vert_grid = np.array(vert_grid, dtype=np.float32)
    tri_grid = np.array(tri_grid, dtype=np.uint32)

    state.sensor_vert_grid = vert_grid
    state.sensor_tri_grid = tri_grid


def crop_center_image(img, cropx, cropy):
    y, x, nb = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    cropped = []
    for bn in range(nb):
        cropped.append(img[starty : starty + cropy, startx : startx + cropx, bn])

    return np.dstack(cropped)


def load_sensor_texture(state: State, impix):
    if not state.sensor_isRGB:
        impix = crop_center_image(impix, state.sensor_tex_width, state.sensor_tex_height)
    state.sensor_tex_pixels = impix


###################
## Rendering Params and math helpers
###################


def init_render_params(state: State):
    xmin, ymin, xmax, ymax = state.bounding_box_estimate

    box_size_x = xmax - xmin
    box_size_y = ymax - ymin

    state.image_width = int(box_size_x / state.image_gsd)
    state.image_height = int(box_size_y / state.image_gsd)

    state.image_corner_coords = np.zeros(2)
    state.image_corner_coords[0] = xmin - state.terrain_xmin
    state.image_corner_coords[1] = ymin - state.terrain_ymin

    state.image_geot = np.zeros(6)
    state.image_geot[0] = xmin + state.terrain_gsd / 2.0
    state.image_geot[3] = ymax - state.terrain_gsd / 2.0

    state.image_geot[1] = state.image_gsd
    state.image_geot[5] = -1.0 * state.image_gsd

    state.image_geot[2] = 0.0
    state.image_geot[4] = 0.0

    # TODO fix in case these are different
    state.image_wkt = state.terrain_wkt

    state.sensor_pos = np.array(
        [
            state.utm_sensor_pos[0] - state.terrain_xmin,
            state.utm_sensor_pos[1] - state.terrain_ymin,
            state.utm_sensor_pos[2],
        ]
    )

    cross_p = np.cross(state.sensor_up, state.sensor_dir)
    state.sensor_right = cross_p / np.linalg.norm(cross_p)

    # need to define focal length in our virtual camera such that we maximize the
    # focal plane size - this gives us better accuracy in our hit locations
    # so subtract z location of sensor from max terrain height as an estimate
    # TODO more elegant way to slide this down towards terrain
    state.sensor_focal_length = (state.utm_sensor_pos[2] - state.terrain_zmax) * 0.5
    state.sensor_focal_plane_size = np.array(
        [
            state.sensor_focal_length * 2.0 * np.tan(state.sensor_hor_fov / 2.0),
            state.sensor_focal_length * 2.0 * np.tan(state.sensor_ver_fov / 2.0),
        ]
    )

    state.sensor_focal_plane_center = (
        state.sensor_pos - state.sensor_dir * state.sensor_focal_length
    )
    state.half_focal_plane_x = state.sensor_right * state.sensor_focal_plane_size[0] * 0.5
    state.half_focal_plane_y = state.sensor_up * state.sensor_focal_plane_size[1] * 0.5
    state.sensor_focal_plane_origin = (
        state.sensor_focal_plane_center - state.half_focal_plane_x - state.half_focal_plane_y
    )


###################
## Optix Setup
###################
class Logger:
    def __init__(self):
        self.num_mssgs = 0

    def __call__(self, level, tag, mssg):
        print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))
        self.num_mssgs += 1


def log_callback(level, tag, mssg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))


def init_optix():
    print("Initializing cuda ...")
    cp.cuda.runtime.free(0)

    print("Initializing optix ...")
    optix.init()


def round_up(val, mult_of):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of


def get_aligned_itemsize(formats, alignment):
    names = []
    for i in range(len(formats)):
        names.append("x" + str(i))

    temp_dtype = np.dtype({"names": names, "formats": formats, "align": True})
    return round_up(temp_dtype.itemsize, alignment)


def optix_version_gte(version):
    if optix.version()[0] > version[0]:
        return True
    if optix.version()[0] == version[0] and optix.version()[1] >= version[1]:
        return True
    return False


def array_to_device_memory(numpy_array, stream=cp.cuda.Stream()):
    byte_size = numpy_array.size * numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p(numpy_array.ctypes.data)
    d_mem = cp.cuda.memory.alloc(byte_size)
    d_mem.copy_from_async(h_ptr, byte_size, stream)
    return d_mem


def create_ctx():
    print("Creating optix device context ...")

    # Note that log callback data is no longer needed.  We can
    # instead send a callable class instance as the log-function
    # which stores any data needed
    global logger
    logger = Logger()

    # OptiX param struct fields can be set with optional
    # keyword constructor arguments.
    ctx_options = optix.DeviceContextOptions(logCallbackFunction=logger, logCallbackLevel=4)

    # They can also be set and queried as properties on the struct
    if optix.version()[1] >= 2:
        ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL

    cu_ctx = 0
    return optix.deviceContextCreate(cu_ctx, ctx_options)


def compile_cuda(
    cuda_file, cuda_tk_path, include_path, project_include_path, stddef_path="/usr/include/linux"
):
    with open(cuda_file, "rb") as f:
        src = f.read()
    nvrtc_dll = os.environ.get("NVRTC_DLL")
    if nvrtc_dll is None:
        nvrtc_dll = ""
    # print("NVRTC_DLL = {}".format(nvrtc_dll))
    prog = Program(src.decode(), cuda_file, lib_name=nvrtc_dll)
    compile_options = [
        "-use_fast_math",
        "-lineinfo",
        "-default-device",
        "-std=c++11",
        "-rdc",
        "true",
        #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
        f"-I{cuda_tk_path}",
        f"-I{include_path}",
        f"-I{project_include_path}",
    ]
    # Optix 7.0 compiles need path to system stddef.h
    # the value of optix.stddef_path is compiled in constant. When building
    # the module, the value can be specified via an environment variable, e.g.
    #   export PYOPTIX_STDDEF_DIR="/usr/include/linux"
    if optix.version()[1] == 0:
        compile_options.append(f"-I{stddef_path}")

    ptx = prog.compile(compile_options)
    return ptx


def set_pipeline_options():
    if optix.version()[1] >= 2:
        return optix.PipelineCompileOptions(
            usesMotionBlur=False,
            traversableGraphFlags=int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
            numPayloadValues=3,
            numAttributeValues=3,
            exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
            pipelineLaunchParamsVariableName="params",
            usesPrimitiveTypeFlags=optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE,
        )
    else:
        return optix.PipelineCompileOptions(
            usesMotionBlur=False,
            traversableGraphFlags=int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS),
            numPayloadValues=3,
            numAttributeValues=3,
            exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
            pipelineLaunchParamsVariableName="params",
        )


def create_module(ctx, pipeline_options, triangle_ptx):
    print("Creating optix module ...")

    module_options = optix.ModuleCompileOptions(
        maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
    )

    module, log = ctx.moduleCreateFromPTX(module_options, pipeline_options, triangle_ptx)
    print("\tModule create log: <<<{}>>>".format(log))
    return module


def create_program_groups(ctx, module):
    print("Creating program groups ... ")

    raygen_prog_group_desc = optix.ProgramGroupDesc()
    raygen_prog_group_desc.raygenModule = module
    raygen_prog_group_desc.raygenEntryFunctionName = "__raygen__rg"
    raygen_prog_group, log = ctx.programGroupCreate([raygen_prog_group_desc])
    print("\tProgramGroup raygen create log: <<<{}>>>".format(log))

    miss_prog_group_desc = optix.ProgramGroupDesc()
    miss_prog_group_desc.missModule = module
    miss_prog_group_desc.missEntryFunctionName = "__miss__ms"
    optix.ProgramGroupOptions()
    miss_prog_group, log = ctx.programGroupCreate([miss_prog_group_desc])
    print("\tProgramGroup miss create log: <<<{}>>>".format(log))

    hitgroup_prog_group_desc = optix.ProgramGroupDesc()
    hitgroup_prog_group_desc.hitgroupModuleCH = module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__terrain_ch"
    hitgroup_prog_group, log = ctx.programGroupCreate([hitgroup_prog_group_desc])
    print("\tProgramGroup hitgroup create log: <<<{}>>>".format(log))

    return [raygen_prog_group, miss_prog_group, hitgroup_prog_group]


def create_pipeline(ctx, program_groups, pipeline_compile_options):
    print("Creating pipeline ... ")

    max_trace_depth = 2
    pipeline_link_options = optix.PipelineLinkOptions()
    pipeline_link_options.maxTraceDepth = max_trace_depth
    pipeline_link_options.debugLevel = optix.COMPILE_DEBUG_LEVEL_MODERATE

    log = ""
    pipeline = ctx.pipelineCreate(
        pipeline_compile_options, pipeline_link_options, program_groups, log
    )
    print("pipeline create log: " + log)

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes(prog_group, stack_sizes)

    (
        dc_stack_size_from_trav,
        dc_stack_size_from_state,
        cc_stack_size,
    ) = optix.util.computeStackSizes(
        stack_sizes, max_trace_depth, 0, 0  # maxCCDepth  # maxDCDepth
    )

    pipeline.setStackSize(
        dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size, 1  # maxTraversableDepth
    )

    return pipeline


def create_sbt(prog_groups):
    # print( "Creating sbt ... " )

    (raygen_prog_group, miss_prog_group, hitgroup_prog_group) = prog_groups

    global d_raygen_sbt
    global d_miss_sbt

    header_format = "{}B".format(optix.SBT_RECORD_HEADER_SIZE)

    #
    # raygen record
    #
    formats = [header_format]
    itemsize = get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype({"names": ["header"], "formats": formats, "itemsize": itemsize, "align": True})
    h_raygen_sbt = np.array([0], dtype=dtype)
    optix.sbtRecordPackHeader(raygen_prog_group, h_raygen_sbt)
    global d_raygen_sbt
    d_raygen_sbt = array_to_device_memory(h_raygen_sbt)

    #
    # miss record
    #
    formats = [header_format, "f4", "f4", "f4"]
    itemsize = get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype(
        {
            "names": ["header", "r", "g", "b"],
            "formats": formats,
            "itemsize": itemsize,
            "align": True,
        }
    )
    h_miss_sbt = np.array([(0, 0.3, 0.1, 0.2)], dtype=dtype)
    # h_miss_sbt = np.array( [ (0, 0.0, 0.0, 0.0) ], dtype=dtype )
    optix.sbtRecordPackHeader(miss_prog_group, h_miss_sbt)
    d_miss_sbt = array_to_device_memory(h_miss_sbt)

    #
    # hitgroup record
    #
    formats = [header_format]
    itemsize = get_aligned_itemsize(formats, optix.SBT_RECORD_ALIGNMENT)
    dtype = np.dtype({"names": ["header"], "formats": formats, "itemsize": itemsize, "align": True})
    h_hitgroup_sbt = np.array([(0)], dtype=dtype)
    optix.sbtRecordPackHeader(hitgroup_prog_group, h_hitgroup_sbt)
    global d_hitgroup_sbt
    d_hitgroup_sbt = array_to_device_memory(h_hitgroup_sbt)

    return optix.ShaderBindingTable(
        raygenRecord=d_raygen_sbt.ptr,
        missRecordBase=d_miss_sbt.ptr,
        missRecordStrideInBytes=h_miss_sbt.dtype.itemsize,
        missRecordCount=1,
        hitgroupRecordBase=d_hitgroup_sbt.ptr,
        hitgroupRecordStrideInBytes=h_hitgroup_sbt.dtype.itemsize,
        hitgroupRecordCount=1,
    )


def create_accel(state: State, ctx, cuda_stream):
    vert_grid_ter = state.terrain_vert_grid
    vert_grid_sen = state.sensor_vert_grid

    vert_grid = np.concatenate([vert_grid_ter, vert_grid_sen])

    # update inds in sensor
    tri_grid_sen = state.sensor_tri_grid
    tri_grid_sen = tri_grid_sen + int(vert_grid_ter.size / 3)

    tri_grid_ter = state.terrain_tri_grid
    tri_grid = np.concatenate([tri_grid_ter, tri_grid_sen])

    accel_options = optix.AccelBuildOptions(
        buildFlags=int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS),
        operation=optix.BUILD_OPERATION_BUILD,
    )

    d_vert_grid = cp.array(vert_grid, dtype="f4")
    nverts = int(len(d_vert_grid) / 3)

    d_tri_grid = cp.array(tri_grid, dtype=np.uint32)
    n_ind_triplets = int(len(d_tri_grid) / 3)

    ortho_input_flags = [optix.GEOMETRY_FLAG_NONE]
    ortho_input = optix.BuildInputTriangleArray()
    ortho_input.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    ortho_input.numVertices = nverts
    ortho_input.vertexBuffers = [d_vert_grid.data.ptr]

    ortho_input.indexFormat = optix.INDICES_FORMAT_UNSIGNED_INT3
    ortho_input.numIndexTriplets = n_ind_triplets
    ortho_input.indexBuffer = d_tri_grid.data.ptr
    ortho_input.indexStrideInBytes = 0

    ortho_input.flags = ortho_input_flags
    ortho_input.numSbtRecords = 1

    gas_buffer_sizes = ctx.accelComputeMemoryUsage([accel_options], [ortho_input])

    d_temp_buffer_gas = cp.cuda.alloc(gas_buffer_sizes.tempSizeInBytes)
    d_gas_output_buffer = cp.cuda.alloc(gas_buffer_sizes.outputSizeInBytes)

    gas_handle = ctx.accelBuild(
        cuda_stream.ptr,  # CUDA stream
        # 0,
        [accel_options],
        [ortho_input],
        d_temp_buffer_gas.ptr,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer.ptr,
        gas_buffer_sizes.outputSizeInBytes,
        [],  # emitted properties
    )
    # cuda_stream.synchronize()

    return (gas_handle, d_gas_output_buffer)


def launch(state: State, pipeline, sbt, trav_handle, cuda_stream):
    d_tex_pix = cp.array(state.sensor_tex_pixels)
    alpha = state.sensor_alpha
    d_tex_pix = cp.dstack([d_tex_pix, alpha])

    tex_height, tex_width, tex_chan = d_tex_pix.shape
    d_tex_pix = cp.reshape(d_tex_pix, (tex_height, tex_chan * tex_width))

    chan_desc = cp.cuda.texture.ChannelFormatDescriptor(
        8, 8, 8, 8, cp.cuda.runtime.cudaChannelFormatKindUnsigned
    )

    d_cuda_arr = cp.cuda.texture.CUDAarray(chan_desc, tex_width, tex_height)

    d_cuda_arr.copy_from(d_tex_pix, stream=cuda_stream)
    # cuda_stream.synchronize()

    res_desc = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, d_cuda_arr)

    tex_desc = cp.cuda.texture.TextureDescriptor(
        addressModes=[cp.cuda.runtime.cudaAddressModeBorder, cp.cuda.runtime.cudaAddressModeBorder],
        filterMode=cp.cuda.runtime.cudaFilterModePoint,
        readMode=cp.cuda.runtime.cudaReadModeElementType,
        normalizedCoords=1,
    )

    pix_width = state.image_width
    pix_height = state.image_height
    d_pix = state.image

    float2_dtype = np.dtype([("x", "f4"), ("y", "f4")], align=True)

    tex_obj = cp.cuda.texture.TextureObject(res_desc, tex_desc)

    # TODO remove the padding variable after data alignment is addressed
    params = [
        ("u8", "trav_handle", trav_handle),
        ("u8", "sensor_tex", tex_obj.ptr),
        ("u8", "image", d_pix.data.ptr),
        ("u4", "image_width", pix_width),
        ("u4", "image_height", pix_height),
        ("f4", "image_corner_coords_x", state.image_corner_coords[0].astype(np.float32)),
        ("f4", "image_corner_coords_y", state.image_corner_coords[1].astype(np.float32)),
        ("f4", "image_gsd", state.image_gsd.astype(np.float32)),
        ("f4", "sensor_focal_length", state.sensor_focal_length.astype(np.float32)),
        ("f4", "terrain_zmax", state.terrain_zmax.astype(np.float32)),
        ("f4", "sensor_pos_x", state.sensor_pos[0].astype(np.float32)),
        ("f4", "sensor_pos_y", state.sensor_pos[1].astype(np.float32)),
        ("f4", "sensor_pos_z", state.sensor_pos[2].astype(np.float32)),
        ("f4", "sensor_up_x", state.sensor_up[0].astype(np.float32)),
        ("f4", "sensor_up_y", state.sensor_up[1].astype(np.float32)),
        ("f4", "sensor_up_z", state.sensor_up[2].astype(np.float32)),
        ("f4", "sensor_right_x", state.sensor_right[0].astype(np.float32)),
        ("f4", "sensor_right_y", state.sensor_right[1].astype(np.float32)),
        ("f4", "sensor_right_z", state.sensor_right[2].astype(np.float32)),
        (
            "f4",
            "sensor_focal_plane_origin_x",
            state.sensor_focal_plane_origin[0].astype(np.float32),
        ),
        (
            "f4",
            "sensor_focal_plane_origin_y",
            state.sensor_focal_plane_origin[1].astype(np.float32),
        ),
        (
            "f4",
            "sensor_focal_plane_origin_z",
            state.sensor_focal_plane_origin[2].astype(np.float32),
        ),
        ("f4", "DUMMY", 1.0),
        (
            float2_dtype,
            "sensor_focal_plane_size",
            (
                state.sensor_focal_plane_size[0].astype(np.float32),
                state.sensor_focal_plane_size[1].astype(np.float32),
            ),
        ),
    ]

    formats = [x[0] for x in params]
    names = [x[1] for x in params]
    values = [x[2] for x in params]

    itemsize = get_aligned_itemsize(formats, 8)
    params_dtype = np.dtype(
        {"names": names, "formats": formats, "itemsize": itemsize, "align": True}
    )

    h_params = np.array([tuple(values)], dtype=params_dtype)
    d_params = array_to_device_memory(h_params, stream=cuda_stream)
    # cuda_stream.synchronize()

    optix.launch(
        pipeline,
        cuda_stream.ptr,
        d_params.ptr,
        h_params.dtype.itemsize,
        sbt,
        pix_width,
        pix_height,
        10,  # depth
    )
    cuda_stream.synchronize()

    d_pix = cp.reshape(d_pix, (state.image_height, state.image_width, 4))
    d_pix = cp.flipud(d_pix)
    h_pix = cp.asnumpy(d_pix)

    return h_pix
