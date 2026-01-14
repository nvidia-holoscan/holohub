"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import re

import numpy as np
from convert_gxf_entities_to_images import convert_gxf_entity_to_images
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))


def check_frames(src_dir, val_dir, threshold=0.005):
    is_valid = True
    count = 0

    while True:
        count += 1

        src_img_path = src_dir + str(count).zfill(4) + ".png"
        val_img_path = val_dir + str(count).zfill(4) + ".png"

        if not (os.path.exists(src_img_path) and os.path.exists(val_img_path)):
            print("End of available frames")
            break

        src_img_arr = np.asarray(Image.open(src_img_path)).astype(int)
        val_img_arr = np.asarray(Image.open(val_img_path)).astype(int)

        if src_img_arr.shape != val_img_arr.shape:
            print("Frames are a different size")
            is_valid = False
            break

        img_diff = np.sum(np.abs(src_img_arr - val_img_arr))

        max_diff = 255 * len(src_img_arr) * len(src_img_arr[0]) * len(src_img_arr[0][0])

        threshold_value = (
            max_diff * threshold
        )  # creates a configurable pixel difference threshold for the frames to match

        print(f"Checking frame({count}) difference: {100 * img_diff / max_diff:.3f}%")

        if img_diff > threshold_value:
            print("Frames exceed threshold for difference")
            is_valid = False

    if count <= 10:
        is_valid = False

    return is_valid


def main():
    parser = argparse.ArgumentParser(
        description=("Command line utility for comparing raw video frames between GXF Tensors")
    )
    parser.add_argument(
        "--source_video_dir",
        default="./recording_output/",
        help="Directory for the source GXF files to compare",
    )
    parser.add_argument(
        "--source_video_basename",
        default="source",
        help="Basename for the source GXF files to compare",
    )
    parser.add_argument(
        "--output_dir",
        default="./recording_output/",
        help="Directory for the output source frames to be stored",
    )
    parser.add_argument(
        "--validation_frames_dir",
        default="./",
        help="Directory for the validation frames to compare",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Threshold for pixel difference between 0 and 1 (default: 0.05)",
    )
    args = parser.parse_args()

    # clean existing frames from output dir
    for f in os.listdir(args.output_dir):
        if re.search("source[0-9]{4}.png", f):
            print("Removing", f)
            os.remove(os.path.join(args.output_dir, f))

    # source to frames
    convert_gxf_entity_to_images(
        args.source_video_dir, args.source_video_basename, args.output_dir, "source"
    )

    valid_output = check_frames(
        args.output_dir + "/source", args.validation_frames_dir, args.threshold
    )

    if valid_output:
        print("Valid video output!")
    else:
        print("Invalid video output!")


if __name__ == "__main__":
    main()
