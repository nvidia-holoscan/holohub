#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

import cupy as cp


def get_gpu_info():
    """
    Retrieves GPU information including name, compute capability, and SM count.

    Returns:
        list: A list of formatted GPU information strings.
    """
    gpu_info_list = []
    device_count = cp.cuda.runtime.getDeviceCount()

    if device_count == 0:
        print("No NVIDIA GPUs detected.", file=sys.stderr)
        return gpu_info_list

    for device_id in range(device_count):
        # Retrieve device properties
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props["name"].decode("utf-8").replace(" ", "-")
        compute_capability = f"c{props['major']}{props['minor']}"
        sm_count = f"n{props['multiProcessorCount']}"

        # Format the GPU information string
        gpu_info = f"{name}_{compute_capability}_{sm_count}"
        gpu_info_list.append(gpu_info)

    return gpu_info_list


def convert_onnx(input_file, output_file, fp16_enabled):
    """
    Convert ONNX model to TensorRT engine using trtexec command.

    Args:
        input_file (str): Path to the input ONNX file.
        output_file (str): Path to the output engine file.
        fp16_enabled (bool): Flag indicating whether to enable FP16 mode.

    Returns:
        None
    """
    trtexec_cmd = f"trtexec --onnx='{input_file}' --saveEngine='{output_file}'"
    if fp16_enabled:
        trtexec_cmd += " --fp16"
    status = os.system(trtexec_cmd)
    return os.WEXITSTATUS(status)


def main():
    """
    Main function to parse command line arguments and convert ONNX model to TensorRT engine.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--fp16", default=False, action="store_true", help="Enable FP16 mode")
    parser.add_argument(
        "--force", default=False, action="store_true", help="Force overwrite existing output file"
    )

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    if os.path.isdir(output_file):
        gpus = get_gpu_info()
        if not gpus:
            print("detected no GPUs, exiting", file=sys.stderr)
            return
        output_file = os.path.join(output_file, f"{gpus[0]}.engine")
    if output_file and os.path.exists(output_file) and not args.force:
        print(
            f"Output file ('{output_file}') already exists. Use --force if you want"
            " to overwrite it.",
            file=sys.stderr,
        )
        return
    fp16_enabled = args.fp16

    result_code = convert_onnx(input_file, output_file, fp16_enabled)
    if result_code != 0:
        print("TensorRT engine generation failed.", file=sys.stderr)
        print("Exit code is ", result_code)
        sys.exit(result_code)
    print(f"TensorRT engine saved to '{output_file}'.")


if __name__ == "__main__":
    main()

# Example
# ./generate_trt_engine.py --input tool_loc_convlstm.onnx --fp16 --output ./ --force
