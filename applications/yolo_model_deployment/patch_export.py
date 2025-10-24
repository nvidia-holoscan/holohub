#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

# Read the file
with open("YOLOv8-TensorRT/export-det.py", "r") as f:
    content = f.read()

# Adds `dynamo=False` to the `torch.onnx.export` call to force the legacy
# TorchScript exporter, as the default Dynamo backend produces incorrect
# results or fails with YOLOv8 models.
content = re.sub(
    r"(output_names=\['num_dets', 'bboxes', 'scores', 'labels'\])\)",
    r"\1,\n            dynamo=False)",
    content,
)

# Write back
with open("YOLOv8-TensorRT/export-det.py", "w") as f:
    f.write(content)

print("Patched export-det.py to use dynamo=False")
