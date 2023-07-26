# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import fractions
import logging
import time
from threading import Condition, Event
from typing import Tuple

import cupy as cp
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from av import VideoFrame
from holoscan.core import Operator, OperatorSpec


class VideoStreamTrack(MediaStreamTrack):
    """
    This class represents a single video track within a stream.

    In the recv function called by aiortc it waits for video frames provided by the
    WebRTCServerOp and returns them.
    """

    kind = "video"

    def __init__(
        self, video_frame_available: Condition, video_frame_consumed: Condition, video_frames: list
    ):
        super().__init__()
        self._video_frame_available = video_frame_available
        self._video_frame_consumed = video_frame_consumed
        self._video_frames = video_frames
        self._timestamp = 0
        self._first_frame = True

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        VIDEO_TIME_BASE = fractions.Fraction(1, 1000)  # ms

        if self._first_frame:
            self._start = time.time()
            self._first_frame = False
        else:
            self._timestamp = int((time.time() - self._start) / VIDEO_TIME_BASE)
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self):
        frame = None
        with self._video_frame_available:
            while not self._video_frames:
                self._video_frame_available.wait()

            pts, time_base = await self.next_timestamp()
            frame = self._video_frames.pop(0)

        with self._video_frame_consumed:
            self._video_frame_consumed.notify_all()

        frame.pts = pts
        frame.time_base = time_base

        return frame


class WebRTCServerOp(Operator):
    def __init__(self, *args, **kwargs):
        self._connected = False
        self._connected_event = Event()
        self._video_frame_available = Condition()
        self._video_frame_consumed = Condition()
        self._video_frames = []
        self._pcs = set()
        super().__init__(*args, **kwargs)

    async def offer(self, sdp, type):
        offer = RTCSessionDescription(sdp, type)

        pc = RTCPeerConnection()
        self._pcs.add(pc)

        pc.addTrack(
            VideoStreamTrack(
                self._video_frame_available, self._video_frame_consumed, self._video_frames
            )
        )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info(f"Connection state {pc.connectionState}")
            if pc.connectionState == "connected":
                self._connected = True
                self._connected_event.set()
            elif pc.connectionState == "failed":
                await pc.close()
                self._pcs.discard(pc)
                self._connected = False
                self._connected_event.set()

        # handle offer
        await pc.setRemoteDescription(offer)

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return (pc.localDescription.sdp, pc.localDescription.type)

    async def shutdown(self):
        # close peer connections
        coros = [pc.close() for pc in self._pcs]
        await asyncio.gather(*coros)
        self._pcs.clear()

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def start(self):
        self._connected_event.wait()
        if not self._connected:
            exit(-1)

    def stop(self):
        pass

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        if isinstance(message, np.ndarray):
            frame = message
        elif isinstance(message, dict):
            tensor = next(iter(message.values()))
            # convert tensor to numpy
            frame = cp.asnumpy(cp.asarray(tensor))
        else:
            raise Exception("Unexpected type ", type(message))

        # wait for the previous frame to be consumed
        with self._video_frame_consumed:
            while self._video_frames:
                self._video_frame_consumed.wait()

        # append the new frame and notify the consumer
        with self._video_frame_available:
            self._video_frames.append(VideoFrame.from_ndarray(frame, format="rgb24"))
            self._video_frame_available.notify_all()
