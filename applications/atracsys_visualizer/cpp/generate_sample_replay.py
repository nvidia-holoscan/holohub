#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

ENTITY_INDEX_STRUCT = struct.Struct("=QQQ")
ENTITY_HEADER_STRUCT = struct.Struct("=QIQIQQ")
COMPONENT_HEADER_STRUCT = struct.Struct("=QQQQ")
TENSOR_HEADER_STRUCT = struct.Struct("=iiQI8i8Q")

SHAPE_MAX_RANK = 8
MEMORY_STORAGE_DEVICE = 1
PRIMITIVE_UNSIGNED8 = 2
PRIMITIVE_FLOAT32 = 9
TENSOR_TYPE = (3996102265592038524, 11968035723744066232)


@dataclass(frozen=True)
class TensorPayload:
    data: bytes
    bytes_per_element: int
    dims: tuple[int, ...]
    strides: tuple[int, ...]
    primitive_type: int

    @property
    def rank(self) -> int:
        return len(self.dims)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny Atracsys replay sample.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-recordings-dir", default="")
    parser.add_argument("--frames", type=int, default=6)
    return parser.parse_args()


def copy_trimmed_recording(source_dir: Path, output_dir: Path, basename: str, frames: int) -> bool:
    index_path = source_dir / f"{basename}.gxf_index"
    entities_path = source_dir / f"{basename}.gxf_entities"
    if not index_path.exists() or not entities_path.exists():
        return False

    with open(index_path, "rb") as src_index, open(entities_path, "rb") as src_entities:
        dst_index_path = output_dir / f"{basename}.gxf_index"
        dst_entities_path = output_dir / f"{basename}.gxf_entities"
        with open(dst_index_path, "wb") as dst_index, open(dst_entities_path, "wb") as dst_entities:
            entity_count = min(frames, os.path.getsize(index_path) // ENTITY_INDEX_STRUCT.size)
            start_time = int(time.time() * 1e9)
            offset = 0
            for i in range(entity_count):
                src_index.seek(i * ENTITY_INDEX_STRUCT.size)
                _, data_size, data_offset = ENTITY_INDEX_STRUCT.unpack(
                    src_index.read(ENTITY_INDEX_STRUCT.size)
                )
                src_entities.seek(data_offset)
                payload = src_entities.read(data_size)
                dst_index.write(
                    ENTITY_INDEX_STRUCT.pack(start_time + i * 100_000_000, len(payload), offset)
                )
                dst_entities.write(payload)
                offset += len(payload)
    return True


def padded_shape(values: tuple[int, ...], fill: int) -> tuple[int, ...]:
    return values + (fill,) * (SHAPE_MAX_RANK - len(values))


def pack_tensor(tensor: TensorPayload) -> bytes:
    tensor_header = TENSOR_HEADER_STRUCT.pack(
        MEMORY_STORAGE_DEVICE,
        tensor.primitive_type,
        tensor.bytes_per_element,
        tensor.rank,
        *padded_shape(tensor.dims, 1),
        *padded_shape(tensor.strides, 0),
    )
    return tensor_header + tensor.data


def pack_component(tensor_name: str, tensor: TensorPayload) -> bytes:
    component_name = tensor_name.encode("utf-8")
    component_header = COMPONENT_HEADER_STRUCT.pack(0, *TENSOR_TYPE, len(component_name))
    return component_header + component_name + pack_tensor(tensor)


def pack_entity(sequence_number: int, tensor_name: str, tensor: TensorPayload) -> bytes:
    component = pack_component(tensor_name, tensor)
    entity_header = ENTITY_HEADER_STRUCT.pack(0, 0, sequence_number, 0, 1, 0)
    return entity_header + component


def write_entities(
    output_dir: Path, basename: str, tensor_name: str, tensors: list[TensorPayload]
) -> None:
    with open(output_dir / f"{basename}.gxf_index", "wb") as index_file, open(
        output_dir / f"{basename}.gxf_entities", "wb"
    ) as entity_file:
        start_time = int(time.time() * 1e9)
        offset = 0
        for i, tensor in enumerate(tensors):
            entity = pack_entity(i, tensor_name, tensor)
            index_file.write(
                ENTITY_INDEX_STRUCT.pack(start_time + i * 100_000_000, len(entity), offset)
            )
            entity_file.write(entity)
            offset += len(entity)


def make_u8_image_tensor(
    width: int, height: int, frame_index: int, invert_x: bool = False
) -> TensorPayload:
    data = bytearray(width * height)
    for y in range(height):
        y_value = (255 * y) // max(height - 1, 1)
        row_offset = y * width
        for x in range(width):
            x_src = width - 1 - x if invert_x else x
            x_value = (255 * x_src) // max(width - 1, 1)
            if invert_x:
                pixel = ((x_value // 2) + (y_value // 2) + frame_index * 6) % 255
            else:
                pixel = (x_value + y_value + frame_index * 10) % 255
            data[row_offset + x] = pixel
    return TensorPayload(
        data=bytes(data),
        bytes_per_element=1,
        dims=(height, width, 1),
        strides=(width, 1, 1),
        primitive_type=PRIMITIVE_UNSIGNED8,
    )


def generate_visible_frames(frames: int) -> list[TensorPayload]:
    return [make_u8_image_tensor(1280, 960, i, invert_x=False) for i in range(frames)]


def generate_ir_frames(frames: int) -> list[TensorPayload]:
    return [make_u8_image_tensor(1280, 960, i, invert_x=True) for i in range(frames)]


def generate_structured_points(frames: int) -> list[TensorPayload]:
    xs = [(-180.0 + (360.0 * x) / 95.0) for x in range(96)]
    ys = [(-120.0 + (240.0 * y) / 71.0) for y in range(72)]
    outputs: list[TensorPayload] = []
    for i in range(frames):
        z_offset = 15.0 * math.sin(i * 0.5)
        values = array("f")
        for y in ys:
            for x in xs:
                values.extend((x, y, 800.0 + z_offset))
        outputs.append(
            TensorPayload(
                data=values.tobytes(),
                bytes_per_element=4,
                dims=(96 * 72, 3),
                strides=(12, 4),
                primitive_type=PRIMITIVE_FLOAT32,
            )
        )
    return outputs


def generate_marker_poses(frames: int) -> list[TensorPayload]:
    outputs: list[TensorPayload] = []
    for i in range(frames):
        tx = 0.02 * math.sin(i * 0.3)
        ty = 0.01 * math.cos(i * 0.3)
        values = array("f", [0.0] * (10 * 16))
        values[0:16] = array(
            "f",
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                tx,
                ty,
                0.8,
                1.0,
            ],
        )
        outputs.append(
            TensorPayload(
                data=values.tobytes(),
                bytes_per_element=4,
                dims=(10, 16),
                strides=(16 * 4, 4),
                primitive_type=PRIMITIVE_FLOAT32,
            )
        )
    return outputs


def generate_synthetic_dataset(output_dir: Path, frames: int) -> None:
    write_entities(output_dir, "visible_base", "base", generate_visible_frames(frames))
    write_entities(output_dir, "ir_base", "base", generate_ir_frames(frames))
    write_entities(
        output_dir,
        "structured_points",
        "structured_points",
        generate_structured_points(frames),
    )
    write_entities(output_dir, "marker_poses", "marker_poses", generate_marker_poses(frames))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source_recordings_dir) if args.source_recordings_dir else None
    basenames = ["visible_base", "ir_base", "structured_points", "marker_poses"]
    if source_dir and source_dir.exists():
        copied = all(
            copy_trimmed_recording(source_dir, output_dir, basename, args.frames)
            for basename in basenames
        )
        if copied:
            print(f"Prepared replay sample by trimming recordings from {source_dir}")
            return 0

    generate_synthetic_dataset(output_dir, args.frames)
    print("Prepared synthetic Atracsys replay sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
