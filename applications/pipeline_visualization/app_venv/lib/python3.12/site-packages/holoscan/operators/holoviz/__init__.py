"""
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

from collections.abc import MutableMapping, Sequence

# BooleanCondition, Allocator, CudaStreamPool are all used as argument types so have to be imported
# before HolovizOp's __init__() can be called.
from holoscan.conditions import BooleanCondition  # noqa: F401
from holoscan.core import IOSpec, io_type_registry
from holoscan.resources import Allocator, CudaStreamPool, UnboundedAllocator  # noqa: F401

from ._holoviz import HolovizOp as _HolovizOp
from ._holoviz import Pose3D  # noqa: F401
from ._holoviz import register_types as _register_types

# register methods for receiving or emitting list[HolovizOp.InputSpec] and camera pose types
_register_types(io_type_registry)

_holoviz_str_to_input_type = {
    "unknown": _HolovizOp.InputType.UNKNOWN,
    "color": _HolovizOp.InputType.COLOR,
    "color_lut": _HolovizOp.InputType.COLOR_LUT,
    "points": _HolovizOp.InputType.POINTS,
    "lines": _HolovizOp.InputType.LINES,
    "line_strip": _HolovizOp.InputType.LINE_STRIP,
    "triangles": _HolovizOp.InputType.TRIANGLES,
    "crosses": _HolovizOp.InputType.CROSSES,
    "rectangles": _HolovizOp.InputType.RECTANGLES,
    "ovals": _HolovizOp.InputType.OVALS,
    "text": _HolovizOp.InputType.TEXT,
    "depth_map": _HolovizOp.InputType.DEPTH_MAP,
    "depth_map_color": _HolovizOp.InputType.DEPTH_MAP_COLOR,
    "points_3d": _HolovizOp.InputType.POINTS_3D,
    "lines_3d": _HolovizOp.InputType.LINES_3D,
    "line_strip_3d": _HolovizOp.InputType.LINE_STRIP_3D,
    "triangles_3d": _HolovizOp.InputType.TRIANGLES_3D,
}

_holoviz_str_to_image_format = {
    "auto_detect": _HolovizOp.ImageFormat.AUTO_DETECT,
    "r8_uint": _HolovizOp.ImageFormat.R8_UINT,
    "r8_sint": _HolovizOp.ImageFormat.R8_SINT,
    "r8_unorm": _HolovizOp.ImageFormat.R8_UNORM,
    "r8_snorm": _HolovizOp.ImageFormat.R8_SNORM,
    "r8_srgb": _HolovizOp.ImageFormat.R8_SRGB,
    "r16_uint": _HolovizOp.ImageFormat.R16_UINT,
    "r16_sint": _HolovizOp.ImageFormat.R16_SINT,
    "r16_unorm": _HolovizOp.ImageFormat.R16_UNORM,
    "r16_snorm": _HolovizOp.ImageFormat.R16_SNORM,
    "r16_sfloat": _HolovizOp.ImageFormat.R16_SFLOAT,
    "r32_uint": _HolovizOp.ImageFormat.R32_UINT,
    "r32_sint": _HolovizOp.ImageFormat.R32_SINT,
    "r32_sfloat": _HolovizOp.ImageFormat.R32_SFLOAT,
    "r8g8b8_unorm": _HolovizOp.ImageFormat.R8G8B8_UNORM,
    "r8g8b8_snorm": _HolovizOp.ImageFormat.R8G8B8_SNORM,
    "r8g8b8_srgb": _HolovizOp.ImageFormat.R8G8B8_SRGB,
    "r8g8b8a8_unorm": _HolovizOp.ImageFormat.R8G8B8A8_UNORM,
    "r8g8b8a8_snorm": _HolovizOp.ImageFormat.R8G8B8A8_SNORM,
    "r8g8b8a8_srgb": _HolovizOp.ImageFormat.R8G8B8A8_SRGB,
    "r16g16b16a16_unorm": _HolovizOp.ImageFormat.R16G16B16A16_UNORM,
    "r16g16b16a16_snorm": _HolovizOp.ImageFormat.R16G16B16A16_SNORM,
    "r16g16b16a16_sfloat": _HolovizOp.ImageFormat.R16G16B16A16_SFLOAT,
    "r32g32b32a32_sfloat": _HolovizOp.ImageFormat.R32G32B32A32_SFLOAT,
    "d16_unorm": _HolovizOp.ImageFormat.D16_UNORM,
    "x8_d24_unorm": _HolovizOp.ImageFormat.X8_D24_UNORM,
    "d32_sfloat": _HolovizOp.ImageFormat.D32_SFLOAT,
    "a2b10g10r10_unorm_pack32": _HolovizOp.ImageFormat.A2B10G10R10_UNORM_PACK32,
    "a2r10g10b10_unorm_pack32": _HolovizOp.ImageFormat.A2R10G10B10_UNORM_PACK32,
    "b8g8r8a8_unorm": _HolovizOp.ImageFormat.B8G8R8A8_UNORM,
    "b8g8r8a8_srgb": _HolovizOp.ImageFormat.B8G8R8A8_SRGB,
    "a8b8g8r8_unorm_pack32": _HolovizOp.ImageFormat.A8B8G8R8_UNORM_PACK32,
    "a8b8g8r8_srgb_pack32": _HolovizOp.ImageFormat.A8B8G8R8_SRGB_PACK32,
    "y8u8y8v8_422_unorm": _HolovizOp.ImageFormat.Y8U8Y8V8_422_UNORM,
    "u8y8v8y8_422_unorm": _HolovizOp.ImageFormat.U8Y8V8Y8_422_UNORM,
    "y8_u8v8_2plane_420_unorm": _HolovizOp.ImageFormat.Y8_U8V8_2PLANE_420_UNORM,
    "y8_u8v8_2plane_422_unorm": _HolovizOp.ImageFormat.Y8_U8V8_2PLANE_422_UNORM,
    "y8_u8_v8_3plane_420_unorm": _HolovizOp.ImageFormat.Y8_U8_V8_3PLANE_420_UNORM,
    "y8_u8_v8_3plane_422_unorm": _HolovizOp.ImageFormat.Y8_U8_V8_3PLANE_422_UNORM,
    "y16_u16v16_2plane_420_unorm": _HolovizOp.ImageFormat.Y16_U16V16_2PLANE_420_UNORM,
    "y16_u16v16_2plane_422_unorm": _HolovizOp.ImageFormat.Y16_U16V16_2PLANE_422_UNORM,
    "y16_u16_v16_3plane_420_unorm": _HolovizOp.ImageFormat.Y16_U16_V16_3PLANE_420_UNORM,
    "y16_u16_v16_3plane_422_unorm": _HolovizOp.ImageFormat.Y16_U16_V16_3PLANE_422_UNORM,
}

_holoviz_str_to_depth_map_render_mode = {
    "points": _HolovizOp.DepthMapRenderMode.POINTS,
    "lines": _HolovizOp.DepthMapRenderMode.LINES,
    "triangles": _HolovizOp.DepthMapRenderMode.TRIANGLES,
}

_holoviz_str_to_yuv_model_conversion = {
    "yuv_601": _HolovizOp.YuvModelConversion.YUV_601,
    "yuv_709": _HolovizOp.YuvModelConversion.YUV_709,
    "yuv_2020": _HolovizOp.YuvModelConversion.YUV_2020,
}

_holoviz_str_to_yuv_range = {
    "itu_full": _HolovizOp.YuvRange.ITU_FULL,
    "itu_narrow": _HolovizOp.YuvRange.ITU_NARROW,
}

_holoviz_str_to_chroma_location = {
    "cosited_even": _HolovizOp.ChromaLocation.COSITED_EVEN,
    "midpoint": _HolovizOp.ChromaLocation.MIDPOINT,
}

_holoviz_str_to_color_space = {
    "srgb_nonlinear": _HolovizOp.ColorSpace.SRGB_NONLINEAR,
    "extended_srgb_linear": _HolovizOp.ColorSpace.EXTENDED_SRGB_LINEAR,
    "bt2020_linear": _HolovizOp.ColorSpace.BT2020_LINEAR,
    "hdr10_st2084": _HolovizOp.ColorSpace.HDR10_ST2084,
    "pass_through": _HolovizOp.ColorSpace.PASS_THROUGH,
    "bt709_linear": _HolovizOp.ColorSpace.BT709_LINEAR,
    "auto": _HolovizOp.ColorSpace.AUTO,
}


class HolovizOp(_HolovizOp):
    def __init__(
        self,
        fragment,
        *args,
        allocator=None,
        receivers=(),
        tensors=(),
        color_lut=(),
        window_title="Holoviz",
        display_name="DP-0",
        width=1920,
        height=1080,
        framerate=60,
        use_exclusive_display=False,
        fullscreen=False,
        headless=False,
        framebuffer_srgb=False,
        vsync=False,
        display_color_space=_HolovizOp.ColorSpace.AUTO,
        enable_render_buffer_input=False,
        enable_render_buffer_output=False,
        enable_depth_buffer_input=False,
        enable_depth_buffer_output=False,
        enable_camera_pose_output=False,
        camera_pose_output_type="projection_matrix",
        camera_eye=(0.0, 0.0, 1.0),
        camera_look_at=(0.0, 0.0, 0.0),
        camera_up=(0.0, 1.0, 0.0),
        key_callback=None,
        unicode_char_callback=None,
        mouse_button_callback=None,
        scroll_callback=None,
        cursor_pos_callback=None,
        framebuffer_size_callback=None,
        window_size_callback=None,
        window_close_callback=None,
        font_path="",
        cuda_stream_pool=None,
        window_close_condition=None,
        name="holoviz_op",
    ):
        if allocator is None:
            allocator = UnboundedAllocator(fragment)

        receiver_iospecs = []
        for receiver in receivers:
            if isinstance(receiver, str):
                continue  # skip
                # raise NotImpelementedError(
                #     "TODO(unknown): need to enable access to self.spec for the OperatorSpec"  # noqa: FIX002
                # )
                # receiver = IOSpec(
                #     op_spec=self.spec,
                #     name=receiver,
                #     io_type=IOSpec.IOType.kInput
                # )
            elif not isinstance(receiver, IOSpec):
                raise ValueError(
                    "receivers must be a string containing the receiver name or an IOSpec object."
                )
            if not receiver.io_type == IOSpec.IOType.kInput:
                raise ValueError("IOType of receiver IOSpec objects must be 'kInput'")
            receiver_iospecs.append(receiver)

        tensor_input_specs = []
        for tensor in tensors:
            # if this already is an InputSpec, just append
            if isinstance(tensor, _HolovizOp.InputSpec):
                tensor_input_specs.append(tensor)
                continue

            # if this is a dict then create an InputSpec by processing the dict entries
            if not isinstance(tensor, MutableMapping):
                raise ValueError(
                    "Tensors must be a sequence of MutableMappings (e.g. list of dict)."
                )
            if "name" not in tensor or not isinstance(tensor["name"], str):
                raise ValueError(
                    "Tensor dictionaries must contain key 'name' with a value that is a str."
                )
            if "type" not in tensor:
                raise ValueError("tensor dictionaries must contain key 'type'")
            valid_keys = {
                "name",
                "type",
                "opacity",
                "priority",
                "image_format",
                "color",
                "line_width",
                "point_size",
                "text",
                "yuv_model_conversion",
                "yuv_range",
                "x_chroma_location",
                "y_chroma_location",
                "depth_map_render_mode",
                "views",
            }
            unrecognized_keys = set(tensor.keys()) - valid_keys
            if unrecognized_keys:
                raise ValueError(f"Unrecognized keys found in tensor: {unrecognized_keys}")
            input_type = tensor["type"]
            ispec = HolovizOp.InputSpec(tensor["name"], tensor["type"])
            ispec.opacity = tensor.get("opacity", 1.0)
            ispec.priority = tensor.get("priority", 0)

            if "image_format" in tensor:
                image_format = tensor["image_format"]
                if isinstance(image_format, str):
                    image_format.lower()
                    if image_format not in _holoviz_str_to_image_format:
                        raise ValueError(f"unrecognized image_format name: {image_format}")
                    image_format = _holoviz_str_to_image_format[image_format]
                elif not isinstance(input_type, _HolovizOp.ImageFormat):
                    raise ValueError(
                        "value corresponding to key 'image_format' must be either a "
                        "HolovizOp.ImageFormat object or one of the following "
                        f"strings: {tuple(_holoviz_str_to_image_format.keys())}"
                    )
                ispec.image_format = image_format

            if "color" in tensor:
                color = tensor["color"]
                if not isinstance(color, Sequence) or len(color) != 4:
                    raise ValueError(
                        "Colors must be specified as a sequence of 4 values: (R, G, B, A)."
                    )
                color = list(map(float, color))
                for val in color:
                    if val < 0.0 or val > 1.0:
                        raise ValueError("color values must be in range [0, 1]")
            else:
                color = (1.0,) * 4
            ispec.color = color
            ispec.line_width = tensor.get("line_width", 1.0)
            ispec.point_size = tensor.get("point_size", 1.0)
            if "text" in tensor:
                text = tensor["text"]
                is_seq = isinstance(text, Sequence) and not isinstance(text, str)
                if not (is_seq and all(isinstance(v, str) for v in text)):
                    raise ValueError("The text value of a tensor must be a sequence of strings")
            else:
                text = []
            ispec.text = tensor.get("text", text)

            if "depth_map_render_mode" in tensor:
                depth_map_render_mode = tensor["depth_map_render_mode"]
                if isinstance(depth_map_render_mode, str):
                    depth_map_render_mode.lower()
                    if depth_map_render_mode not in _holoviz_str_to_depth_map_render_mode:
                        raise ValueError(
                            f"unrecognized depth_map_render_mode name: {depth_map_render_mode}"
                        )
                    depth_map_render_mode = _holoviz_str_to_depth_map_render_mode[
                        depth_map_render_mode
                    ]
                elif not isinstance(input_type, _HolovizOp.DepthMapRenderMode):
                    raise ValueError(
                        "value corresponding to key 'depth_map_render_mode' must be either a "
                        "HolovizOp.DepthMapRenderMode object or one of the following "
                        f"strings: {tuple(_holoviz_str_to_depth_map_render_mode.keys())}"
                    )
            else:
                depth_map_render_mode = _HolovizOp.DepthMapRenderMode.POINTS
            ispec.depth_map_render_mode = depth_map_render_mode

            if "yuv_model_conversion" in tensor:
                yuv_model_conversion = tensor["yuv_model_conversion"]
                if isinstance(yuv_model_conversion, str):
                    yuv_model_conversion.lower()
                    if yuv_model_conversion not in _holoviz_str_to_yuv_model_conversion:
                        raise ValueError(
                            f"unrecognized yuv_model_conversion name: {yuv_model_conversion}"
                        )
                    yuv_model_conversion = _holoviz_str_to_yuv_model_conversion[
                        yuv_model_conversion
                    ]
                elif not isinstance(input_type, _HolovizOp.YuvModelConversion):
                    raise ValueError(
                        "value corresponding to key 'yuv_model_conversion' must be either a "
                        "HolovizOp.YuvModelConversion object or one of the following "
                        f"strings: {tuple(_holoviz_str_to_yuv_model_conversion.keys())}"
                    )
            else:
                yuv_model_conversion = _HolovizOp.YuvModelConversion.YUV_601
            ispec.yuv_model_conversion = yuv_model_conversion

            if "yuv_range" in tensor:
                yuv_range = tensor["yuv_range"]
                if isinstance(yuv_range, str):
                    yuv_range.lower()
                    if yuv_range not in _holoviz_str_to_yuv_range:
                        raise ValueError(f"unrecognized yuv_range name: {yuv_range}")
                    yuv_range = _holoviz_str_to_yuv_range[yuv_range]
                elif not isinstance(input_type, _HolovizOp.YuvRange):
                    raise ValueError(
                        "value corresponding to key 'yuv_range' must be either a "
                        "HolovizOp.YuvRange object or one of the following "
                        f"strings: {tuple(_holoviz_str_to_yuv_range.keys())}"
                    )
            else:
                yuv_range = _HolovizOp.YuvRange.ITU_FULL
            ispec.yuv_range = yuv_range

            if "x_chroma_location" in tensor:
                x_chroma_location = tensor["x_chroma_location"]
                if isinstance(x_chroma_location, str):
                    x_chroma_location.lower()
                    if x_chroma_location not in _holoviz_str_to_chroma_location:
                        raise ValueError(
                            f"unrecognized x_chroma_location name: {x_chroma_location}"
                        )
                    x_chroma_location = _holoviz_str_to_chroma_location[x_chroma_location]
                elif not isinstance(input_type, _HolovizOp.ChromaLocation):
                    raise ValueError(
                        "value corresponding to key 'x_chroma_location' must be either a "
                        "HolovizOp.ChromaLocation object or one of the following "
                        f"strings: {tuple(_holoviz_str_to_chroma_location.keys())}"
                    )
            else:
                x_chroma_location = _HolovizOp.ChromaLocation.COSITED_EVEN
            ispec.x_chroma_location = x_chroma_location

            if "y_chroma_location" in tensor:
                y_chroma_location = tensor["y_chroma_location"]
                if isinstance(y_chroma_location, str):
                    y_chroma_location.lower()
                    if y_chroma_location not in _holoviz_str_to_chroma_location:
                        raise ValueError(
                            f"unrecognized y_chroma_location name: {y_chroma_location}"
                        )
                    y_chroma_location = _holoviz_str_to_chroma_location[y_chroma_location]
                elif not isinstance(input_type, _HolovizOp.ChromaLocation):
                    raise ValueError(
                        "value corresponding to key 'y_chroma_location' must be either a "
                        "HolovizOp.ChromaLocation object or one of the following "
                        f"strings: {tuple(_holoviz_str_to_chroma_location.keys())}"
                    )
            else:
                y_chroma_location = _HolovizOp.ChromaLocation.COSITED_EVEN
            ispec.y_chroma_location = y_chroma_location

            ispec.views = tensor.get("views", [])

            tensor_input_specs.append(ispec)
        super().__init__(
            fragment,
            *args,
            allocator=allocator,
            receivers=receiver_iospecs,
            tensors=tensor_input_specs,
            color_lut=list(color_lut),
            window_title=window_title,
            display_name=display_name,
            width=width,
            height=height,
            framerate=framerate,
            use_exclusive_display=use_exclusive_display,
            fullscreen=fullscreen,
            headless=headless,
            framebuffer_srgb=framebuffer_srgb,
            vsync=vsync,
            display_color_space=display_color_space,
            enable_render_buffer_input=enable_render_buffer_input,
            enable_render_buffer_output=enable_render_buffer_output,
            enable_depth_buffer_input=enable_depth_buffer_input,
            enable_depth_buffer_output=enable_depth_buffer_output,
            enable_camera_pose_output=enable_camera_pose_output,
            camera_pose_output_type=camera_pose_output_type,
            camera_eye=camera_eye,
            camera_look_at=camera_look_at,
            camera_up=camera_up,
            key_callback=key_callback,
            unicode_char_callback=unicode_char_callback,
            mouse_button_callback=mouse_button_callback,
            scroll_callback=scroll_callback,
            cursor_pos_callback=cursor_pos_callback,
            framebuffer_size_callback=framebuffer_size_callback,
            window_size_callback=window_size_callback,
            window_close_callback=window_close_callback,
            font_path=font_path,
            cuda_stream_pool=cuda_stream_pool,
            window_close_condition=window_close_condition,
            name=name,
        )


# copy docstrings defined in operators_pydoc.hpp
HolovizOp.__doc__ = _HolovizOp.__doc__
HolovizOp.__init__.__doc__ = _HolovizOp.__init__.__doc__

__all__ = ["HolovizOp"]
