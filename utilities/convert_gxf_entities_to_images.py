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

import argparse
import sys

from gxf_entity_codec import EntityReader
from PIL import Image


def iter_output_frames(entity_iter):
    for entity in entity_iter:
        component = entity.components[0]  # assume only one component with a tensor
        tensor = component.tensor
        array = tensor.array
        frame_bytes = array.tobytes()
        yield frame_bytes


def convert_gxf_entity_to_images(entity_dir, entity_basename, output_dir, output_name):
    with EntityReader(directory=entity_dir, basename=entity_basename) as reader:
        frame_shape = reader.get_frame(0).shape
        print(
            f"Frame array shape: {frame_shape[0]}x{frame_shape[1]}x{frame_shape[2]}"
            " (height x width x channels)",
            file=sys.stderr,
        )

        entities = reader.get_entities()
        for i, frame_data in enumerate(iter_output_frames(entities)):
            img = Image.frombytes("RGB", (frame_shape[1], frame_shape[0]), frame_data)
            img.save(str(output_dir + "/" + output_name + "{:04d}.png").format(i + 1))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Command line utility for reading raw video frames in GXF Tensors for "
            "stream recording."
        )
    )
    parser.add_argument("--basename", default="tensor", help="Basename for gxf entities to read")
    parser.add_argument("--directory", default="./", help="Directory for gxf entities to read")
    parser.add_argument("--outputname", default="tensor", help="Output name for images")
    parser.add_argument("--outputdir", default="./", help="Directory for output images")
    args = parser.parse_args()

    convert_gxf_entity_to_images(args.directory, args.basename, args.outputdir, args.outputname)


if __name__ == "__main__":
    main()
