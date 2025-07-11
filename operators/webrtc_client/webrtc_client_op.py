# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
from threading import Condition, Event

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamError, MediaStreamTrack
from holoscan import as_tensor
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity


class VideoStreamReceiverContext:
    def __init__(self):
        self.task = None


class VideoStreamReceiver:
    def __init__(self, video_frame_available: Condition, video_frames: list):
        self._video_frame_available = video_frame_available
        self._video_frames = video_frames
        self._receiver_contexts = {}

    def add_track(self, track: MediaStreamTrack):
        context = VideoStreamReceiverContext()
        context.task = asyncio.ensure_future(self._run_track(track, context))
        self._receiver_contexts[track] = context

    def remove_track(self, track: MediaStreamTrack):
        context = self._receiver_contexts[track]
        if context:
            if context.task is not None:
                context.task.cancel()
                context.task = None
            self._receiver_contexts.pop(track)

    async def stop(self):
        for context in self._receiver_contexts:
            if context.task is not None:
                context.task.cancel()
                context.task = None
        self._receiver_contexts.clear()

    async def _run_track(self, track: MediaStreamTrack, context: VideoStreamReceiverContext):
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return

            with self._video_frame_available:
                self._video_frames.append(frame)
                self._video_frame_available.notify_all()


class WebRTCClientOp(Operator):
    def __init__(self, *args, **kwargs):
        self._connected = False
        self._connected_event = Event()
        self._video_frame_available = Condition()
        self._video_frames = []
        self._pcs = set()
        self._receiver = VideoStreamReceiver(self._video_frame_available, self._video_frames)
        super().__init__(*args, **kwargs)

    async def offer(self, sdp, type):
        offer = RTCSessionDescription(sdp, type)

        pc = RTCPeerConnection()
        self._pcs.add(pc)

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

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                self._receiver.add_track(track)

            @track.on("ended")
            async def on_ended():
                self._receiver.remove_track(track)

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
        spec.output("output")

    def start(self):
        self._connected_event.wait()
        if not self._connected:
            exit(-1)

    def stop(self):
        self._receiver.stop()

    def compute(self, op_input, op_output, context):
        video_frame = None
        with self._video_frame_available:
            while not self._video_frames:
                self._video_frame_available.wait()
            video_frame = self._video_frames.pop(0)

        rgb_frame = video_frame.to_rgb()
        array = rgb_frame.to_ndarray()

        entity = Entity(context)
        entity.add(as_tensor(array), "frame")
        op_output.emit(entity, "output")
