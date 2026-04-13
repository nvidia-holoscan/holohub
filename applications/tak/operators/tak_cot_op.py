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
Holoscan operator for uploading YOLO detections to a TAK server as CoT markers.
"""

import logging
import random
import socket
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Optional

import cupy as cp
import pytak
from holoscan.core import Operator, OperatorSpec

logger = logging.getLogger("tak_cot")


class TakCotOp(Operator):
    """Operator that uploads CoT messages to a TAK server via TCP socket."""

    def __init__(
        self,
        fragment,
        *args,
        tak_host: str,
        tak_port: int = 18088,
        base_lat: float = 28.53830862,
        base_lon: float = -81.37923400,
        marker_type: str = "a-h-A-M-A",
        marker_type_map: Optional[dict] = None,
        update_interval: float = 2.0,
        detector_op=None,
        **kwargs,
    ):
        self.tak_host = tak_host
        self.tak_port = tak_port

        self.base_lat = base_lat
        self.base_lon = base_lon
        self.marker_type = marker_type
        self.marker_type_map = marker_type_map or {}
        self.update_interval = update_interval
        self.detector_op = detector_op

        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.detection_count = 0
        self.last_send_time = 0.0
        self._presence_sent = False
        self._detection_offsets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._max_offsets = 1000

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def start(self):
        if not self.tak_host:
            logger.warning(
                "TAK_HOST not configured; TAK integration disabled. "
                "Set TAK_HOST environment variable to enable."
            )
            return
        logger.info("Connecting to TAK server at %s:%d", self.tak_host, self.tak_port)
        self._connect()
        if not self.connected:
            logger.warning(
                "Could not connect to TAK server at %s:%d; "
                "detections will still be visualized but not uploaded.",
                self.tak_host,
                self.tak_port,
            )

    def _connect(self):
        try:
            self.socket = socket.create_connection((self.tak_host, self.tak_port), timeout=5.0)
            self.connected = True
            logger.info("Connected to TAK server successfully")
        except Exception as e:
            logger.error("Connection error: %s", e)
            self.connected = False
            if self.socket:
                self.socket.close()
                self.socket = None

    def _send_eud_presence(self):
        logger.info("Sending EUD presence CoT to register connection")
        try:
            root = ET.Element("event")
            root.set("version", "2.0")
            root.set("type", "a-f-G-E-V-C")
            root.set("uid", "TAK-DETECTION-SYSTEM")
            root.set("how", "m-g")
            root.set("time", pytak.cot_time())
            root.set("start", pytak.cot_time())
            root.set("stale", pytak.cot_time(3600))

            pt_attr = {
                "lat": str(self.base_lat),
                "lon": str(self.base_lon),
                "hae": "0",
                "ce": "10",
                "le": "10",
            }
            ET.SubElement(root, "point", attrib=pt_attr)

            detail = ET.SubElement(root, "detail")
            contact = ET.SubElement(detail, "contact")
            contact.set("callsign", "TAK Detector")

            takv = ET.SubElement(detail, "takv")
            takv.set("platform", "Python-PyTAK")
            takv.set("version", "1.0.0")
            takv.set("device", "NVIDIA Holoscan")
            takv.set("os", "Linux")

            presence_cot = ET.tostring(root, encoding="unicode")
            self.socket.sendall(presence_cot.encode("utf-8") + b"\n")
            self._presence_sent = True
            logger.info("EUD presence CoT sent successfully")
        except Exception as e:
            logger.error("Failed to send EUD presence: %s", e)

    def _generate_cot_xml(
        self,
        uid: str,
        callsign: str,
        lat: float,
        lon: float,
        stale_minutes: int = 5,
        cot_type: str = "",
    ) -> bytes:
        root = ET.Element("event")
        root.set("version", "2.0")
        root.set("type", cot_type or self.marker_type)
        root.set("uid", uid)
        root.set("how", "m-g")
        root.set("time", pytak.cot_time())
        root.set("start", pytak.cot_time())
        root.set("stale", pytak.cot_time(stale_minutes * 60))

        pt_attr = {
            "lat": str(lat),
            "lon": str(lon),
            "hae": "0",
            "ce": "10",
            "le": "10",
        }
        ET.SubElement(root, "point", attrib=pt_attr)

        # OTS creates markers only when <contact> and <takv> are ABSENT.
        # Including either tag causes OTS to treat the CoT as an EUD instead.
        detail = ET.SubElement(root, "detail")
        remarks = ET.SubElement(detail, "remarks")
        remarks.text = callsign

        return ET.tostring(root, encoding="unicode").encode("utf-8")

    def _send_cot_marker(
        self,
        detection_id: str,
        lat: float,
        lon: float,
        label: str = "Detection",
        cot_type: str = "",
    ):
        if not self.connected or not self.socket:
            return False

        try:
            cot_message = self._generate_cot_xml(detection_id, label, lat, lon, cot_type=cot_type)
            self.socket.sendall(cot_message + b"\n")
            logger.info("Sent CoT marker: %s at (%.6f, %.6f)", label, lat, lon)
            return True
        except Exception as e:
            logger.error("CoT send error: %s", e)
            self.connected = False
            self._connect()
            return False

    def compute(self, op_input, op_output, context):
        entity = op_input.receive("in")

        current_time = time.time()
        if current_time - self.last_send_time < self.update_interval:
            return

        self.last_send_time = current_time

        # Retry connection if start() failed (e.g. TAK server wasn't up yet)
        if not self.connected and self.tak_host:
            self._connect()

        # Send presence lazily — at startup RabbitMQ may not be ready yet,
        # so we defer until compute() is running.
        if not self._presence_sent and self.connected:
            self._send_eud_presence()

        try:
            bbox_tensor = entity.get("bbox")
            if bbox_tensor is None:
                return

            bbox_label_tensor = entity.get("bbox_label")
            if bbox_label_tensor is None:
                return

            bbox_labels = cp.asarray(bbox_label_tensor).get()

            track_ids = None
            track_ids_tensor = entity.get("track_ids")
            if track_ids_tensor is not None:
                track_ids = cp.asarray(track_ids_tensor).get().tolist()

            class_ids = None
            class_ids_tensor = entity.get("class_ids")
            if class_ids_tensor is not None:
                class_ids = cp.asarray(class_ids_tensor).get().tolist()

            label_map = {}
            if self.detector_op is not None:
                label_map = getattr(self.detector_op, "label_map", {})

            num_detections = len(bbox_labels)
            logger.debug("Processing %d detections", num_detections)

            for i in range(num_detections):
                self.detection_count += 1

                cls_name = ""
                if class_ids is not None and i < len(class_ids):
                    cls_id = int(class_ids[i])
                    cls_name = label_map.get(cls_id, f"class_{cls_id}")

                if track_ids is not None and i < len(track_ids):
                    track_id = int(track_ids[i])
                    detection_id = f"{cls_name}-{track_id}" if cls_name else f"Detection-{track_id}"
                    detection_label = (
                        f"{cls_name} {track_id}" if cls_name else f"Detection {track_id}"
                    )
                else:
                    slot_id = (self.detection_count % 100) + 1
                    detection_id = f"{cls_name}-{slot_id}" if cls_name else f"Detection-{slot_id}"
                    detection_label = (
                        f"{cls_name} {slot_id}" if cls_name else f"Detection {slot_id}"
                    )

                # Cache offsets per detection_id so markers don't jump
                if detection_id in self._detection_offsets:
                    self._detection_offsets.move_to_end(detection_id)
                else:
                    # ~150 feet ≈ 46 meters ≈ 0.000405 degrees
                    # Place detections south-east (bottom-right) of the base
                    self._detection_offsets[detection_id] = (
                        random.uniform(-0.000405, 0),
                        random.uniform(0, 0.000405),
                    )
                    if len(self._detection_offsets) > self._max_offsets:
                        self._detection_offsets.popitem(last=False)
                lat_offset, lon_offset = self._detection_offsets[detection_id]

                lat = self.base_lat + lat_offset
                lon = self.base_lon + lon_offset

                cot_type = self.marker_type_map.get(cls_name, "")
                self._send_cot_marker(detection_id, lat, lon, detection_label, cot_type=cot_type)
        except Exception as e:
            logger.error("Error processing detections: %s", e, exc_info=True)

    def stop(self):
        logger.info("Sent %d CoT detections total", self.detection_count)
        if self.socket:
            self.socket.close()
