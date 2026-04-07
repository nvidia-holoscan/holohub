# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Holoscan operator that runs YOLOv8 detection using the provided .pt model.
"""

import json
import logging
import os
import time
from typing import List, Optional

import cupy as cp
import holoscan as hs
import numpy as np
import torch
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import HolovizOp
from ultralytics import YOLO

logger = logging.getLogger("detector")


class DetectorOp(Operator):
    """Run YOLOv8 detections and emit Holoviz-friendly rectangles/specs."""

    def __init__(
        self,
        fragment,
        *args,
        model_path: str,
        confidence: float = 0.5,
        label_map: Optional[dict[int, str]] = None,
        imgsz: Optional[int] = 640,
        letterbox_meta_path: Optional[str] = None,
        bytetrack_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self.model_path = model_path
        self.confidence = confidence
        self.label_map = label_map or {}
        self.imgsz = imgsz
        self.letterbox_meta_path = letterbox_meta_path
        self.bytrack_path = bytetrack_path
        self.letterbox_meta: Optional[dict] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[YOLO] = None

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("output_specs")
        spec.output("outputs")
        spec.output("tak_out")

    def start(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        if self.letterbox_meta_path and os.path.exists(self.letterbox_meta_path):
            try:
                with open(self.letterbox_meta_path, "r", encoding="utf-8") as f:
                    self.letterbox_meta = json.load(f)
            except Exception:
                logger.warning(
                    "Failed to load letterbox metadata from %s, " "bbox correction disabled",
                    self.letterbox_meta_path,
                    exc_info=True,
                )
                self.letterbox_meta = None
        try:
            if not self.label_map:
                names = getattr(self.model, "names", None)
                if isinstance(names, dict):
                    self.label_map = {int(k): v for k, v in names.items()}
                elif isinstance(names, list):
                    self.label_map = {i: n for i, n in enumerate(names)}
                elif names is not None:
                    logger.warning(
                        "Unexpected model.names type %s (value: %r); "
                        "falling back to class_N labels",
                        type(names).__name__,
                        names,
                    )
        except Exception:
            logger.warning(
                "Failed to derive label_map from model.names (value: %r); "
                "falling back to class_N labels",
                getattr(self.model, "names", None),
                exc_info=True,
            )
        # Warmup on small tensor to avoid first-iteration latency.
        imgsz = self.imgsz or 640
        dummy = torch.zeros((1, 3, imgsz, imgsz), device=self.device, dtype=torch.float32)
        _ = self.model.predict(dummy, device=self.device, verbose=False)

    def compute(self, op_input, op_output, context):
        if self.model is None:
            raise RuntimeError("YOLO model not initialized")

        msg = op_input.receive("in")

        tensor = msg.get("", None) or msg.get("tensor", None) or msg.get("source_video")
        if tensor is None:
            raise RuntimeError("No tensor found on message for detector")

        frame = cp.asarray(tensor).get()  # HWC on host
        h, w, _ = frame.shape

        frame_uint8 = np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8, copy=False))

        # Strip alpha channel (e.g. V4L2 RGBA) — model expects 3-channel input
        if frame_uint8.ndim == 3 and frame_uint8.shape[2] == 4:
            frame_uint8 = frame_uint8[..., :3]

        frame_for_model = frame_uint8
        pad_x = pad_y = 0
        content_w = w
        content_h = h
        norm_w = w
        norm_h = h
        if self.letterbox_meta:
            pad_x = int(self.letterbox_meta.get("pad_x", 0) or 0)
            pad_y = int(self.letterbox_meta.get("pad_y", 0) or 0)
            content_w = int(self.letterbox_meta.get("content_width", w) or w)
            content_h = int(self.letterbox_meta.get("content_height", h) or h)
            norm_w = int(self.letterbox_meta.get("output_width", w) or w)
            norm_h = int(self.letterbox_meta.get("output_height", h) or h)
            x1 = min(frame_uint8.shape[1], pad_x + content_w)
            y1 = min(frame_uint8.shape[0], pad_y + content_h)
            x0 = max(0, pad_x)
            y0 = max(0, pad_y)
            frame_for_model = frame_uint8[y0:y1, x0:x1]

        tracker_arg = self.bytrack_path or "bytetrack.yaml"

        infer_t0 = time.perf_counter()
        results = self.model.track(
            frame_for_model,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False,
            tracker=tracker_arg,
        )
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        logger.debug(
            "YOLO inference: %.2f ms on frame %dx%d",
            infer_ms,
            frame_for_model.shape[1],
            frame_for_model.shape[0],
        )

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        confs: List[float] = []
        track_ids: Optional[List[int]] = None
        if results and results[0].boxes is not None:
            b = results[0].boxes
            boxes_xyxy = b.xyxy.cpu().numpy().tolist()
            labels = b.cls.cpu().numpy().astype(int).tolist()
            confs = b.conf.cpu().numpy().tolist()
            track_ids = None
            if b.id is not None:
                track_ids = b.id.cpu().numpy().astype(int).tolist()

        if self.letterbox_meta and boxes_xyxy:
            boxes_xyxy = [
                [x0 + pad_x, y0 + pad_y, x1 + pad_x, y1 + pad_y] for x0, y0, x1, y1 in boxes_xyxy
            ]

        if boxes_xyxy:
            sample = []
            for box, cls_id, conf in zip(boxes_xyxy, labels, confs):
                name = self.label_map.get(cls_id, f"class_{cls_id}")
                sample.append(
                    {
                        "cls": name,
                        "id": int(cls_id),
                        "conf": round(float(conf), 3),
                        "bbox": [round(float(x), 1) for x in box],
                    }
                )
                if len(sample) >= 5:
                    break
            logger.debug(
                "YOLO detections: %d track ids: %s data: %s",
                len(boxes_xyxy),
                track_ids,
                sample,
            )

        specs = []
        entity = Entity(context)

        if not boxes_xyxy:
            op_output.emit(entity, "outputs")
            op_output.emit(specs, "output_specs")
            tak_entity = Entity(context)
            op_output.emit(tak_entity, "tak_out")
            return

        boxes_arr = np.array(boxes_xyxy, dtype=np.float32)
        boxes_arr[:, [0, 2]] /= float(norm_w)
        boxes_arr[:, [1, 3]] /= float(norm_h)
        bboxes_norm = np.reshape(boxes_arr, (1, -1, 2))

        bbox_label = np.asarray([(b[0], max(0.0, b[1] - 0.03)) for b in boxes_arr])
        label_text = []
        for cls_id, conf in zip(labels, confs):
            name = self.label_map.get(cls_id, f"class_{cls_id}")
            label_text.append(f"{name} {conf:.2f}")

        outline_offset = 0.002
        outline_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for i, (dx, dy) in enumerate(outline_dirs):
            key = f"bbox_label_outline_{i}"
            offset_coords = np.asarray(
                [(x + dx * outline_offset, y + dy * outline_offset) for x, y in bbox_label],
                dtype=np.float32,
            )
            entity.add(hs.as_tensor(offset_coords), key)

        entity.add(hs.as_tensor(bboxes_norm), "bbox")
        entity.add(hs.as_tensor(bbox_label), "bbox_label")

        # TAK entity: send track_ids, class_ids, and bbox on a separate port
        # so they don't confuse Holoviz
        tak_entity = Entity(context)
        tak_entity.add(hs.as_tensor(bboxes_norm), "bbox")
        tak_entity.add(hs.as_tensor(bbox_label), "bbox_label")
        if track_ids is not None:
            tak_entity.add(hs.as_tensor(np.array(track_ids, dtype=np.int32)), "track_ids")
        tak_entity.add(hs.as_tensor(np.array(labels, dtype=np.int32)), "class_ids")

        spec_bbox = HolovizOp.InputSpec("bbox", HolovizOp.InputType.RECTANGLES)
        spec_bbox.color = [1.0, 0.2, 0.2, 1.0]
        spec_bbox.line_width = 3

        for i in range(len(outline_dirs)):
            spec_outline = HolovizOp.InputSpec(f"bbox_label_outline_{i}", "text")
            spec_outline.text = label_text or ["Detection"]
            spec_outline.color = [0.0, 0.0, 0.0, 1.0]
            specs.append(spec_outline)

        spec_label = HolovizOp.InputSpec("bbox_label", "text")
        spec_label.text = label_text or ["Detection"]
        spec_label.color = [1.0, 1.0, 1.0, 1.0]

        specs.extend([spec_bbox, spec_label])

        op_output.emit(entity, "outputs")
        op_output.emit(specs, "output_specs")
        op_output.emit(tak_entity, "tak_out")
