#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: prepare_model.sh <output-directory>" >&2
    exit 2
fi

output_directory="$1"
model_directory="${output_directory}/models"
license_directory="${output_directory}/licenses"

depth_anything_revision="a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
tensorrt_export_revision="a5715a72b4ea3320795499a9e0740ff6fe16a180"
model_revision="03876f8651c73a60fe4c2c48294e09fcb6838fcf"
model_sha256="715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"

for command in git wget python3 sha256sum; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "prepare_model.sh: required command not found: ${command}" >&2
        exit 1
    fi
done

if ! python3 -c 'import cv2, onnx, onnxscript, torch' >/dev/null 2>&1; then
    echo "prepare_model.sh: model export requires cv2, onnx, onnxscript, and torch" >&2
    exit 1
fi

mkdir -p "${model_directory}" "${license_directory}"
temporary_directory="$(mktemp -d "${output_directory}/.model-build.XXXXXX")"
trap 'rm -rf -- "${temporary_directory}"' EXIT

git clone --quiet https://github.com/DepthAnything/Depth-Anything-V2.git \
    "${temporary_directory}/Depth-Anything-V2"
git -C "${temporary_directory}/Depth-Anything-V2" checkout --quiet \
    "${depth_anything_revision}"

git clone --quiet https://github.com/spacewalk01/depth-anything-tensorrt.git \
    "${temporary_directory}/depth-anything-tensorrt"
git -C "${temporary_directory}/depth-anything-tensorrt" checkout --quiet \
    "${tensorrt_export_revision}"

cp "${temporary_directory}/depth-anything-tensorrt/depth_anything_v2/dpt.py" \
    "${temporary_directory}/Depth-Anything-V2/depth_anything_v2/dpt.py"
cp "${temporary_directory}/depth-anything-tensorrt/depth_anything_v2/export_v2.py" \
    "${temporary_directory}/Depth-Anything-V2/export_v2.py"
sed -i '/from torchvision.transforms import Compose/d' \
    "${temporary_directory}/Depth-Anything-V2/depth_anything_v2/dpt.py"
sed -i 's/opset_version=11/opset_version=18/' \
    "${temporary_directory}/Depth-Anything-V2/export_v2.py"

mkdir -p "${temporary_directory}/Depth-Anything-V2/checkpoints"
wget -q -O \
    "${temporary_directory}/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth" \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/${model_revision}/depth_anything_v2_vits.pth?download=true"
echo "${model_sha256}  ${temporary_directory}/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth" \
    | sha256sum --check -

(
    cd "${temporary_directory}/Depth-Anything-V2"
    python3 export_v2.py --encoder vits --input-size 518
)

install -m 0644 \
    "${temporary_directory}/Depth-Anything-V2/depth_anything_v2_vits.onnx" \
    "${model_directory}/depth_anything_v2_vits.onnx"
install -m 0644 \
    "${temporary_directory}/Depth-Anything-V2/depth_anything_v2_vits.onnx.data" \
    "${model_directory}/depth_anything_v2_vits.onnx.data"
install -m 0644 \
    "${temporary_directory}/Depth-Anything-V2/LICENSE" \
    "${license_directory}/depth-anything-v2-apache-2.0.txt"
install -m 0644 \
    "${temporary_directory}/depth-anything-tensorrt/LICENSE" \
    "${license_directory}/depth-anything-tensorrt-mit.txt"
