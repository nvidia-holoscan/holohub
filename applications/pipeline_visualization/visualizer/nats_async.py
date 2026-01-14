"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""

import asyncio
import contextlib
import logging
import threading
from enum import Enum
from queue import Full, Queue

import nats

logger = logging.getLogger(__name__)


class _MessageType(Enum):
    """Internal message types for communicating between threads/processes

    Args:
        Enum (int): Type of message
    """

    PUBLISH = 1
    SUBSCRIBE = 2
    UNSUBSCRIBE = 3
    SHUTDOWN = 4


class Publish:
    def __init__(self, subject: str, payload: bytes):
        self.subject = subject
        self.payload = payload
        self.type = _MessageType.PUBLISH


class Unsubscribe:
    def __init__(self, subject: str):
        self.subject = subject
        self.type = _MessageType.UNSUBSCRIBE


class Subscribe:
    def __init__(self, subject: str):
        self.subject = subject
        self.type = _MessageType.SUBSCRIBE


class Shutdown:
    def __init__(self):
        self.type = _MessageType.SHUTDOWN


class _Subscription:
    def __init__(self, nats_subscription: nats.aio.subscription.Subscription, queue: Queue):
        self._nats_subscription = nats_subscription
        self._queue = queue


class NatsAsync:
    def __init__(self, host: str, subscriptions: list[str] | None = None):
        self._host: str = host
        self._initial_subscriptions: list[str] = subscriptions or []
        self._subscriptions: dict[str, _Subscription] = {}
        self._connection: nats.Connection | None = None  # NATS connection
        self._use_js: bool = False
        self._js_context: nats.js.JetStreamContext | None = None
        self._control_queue = Queue()
        self._thread = threading.Thread(target=self._start_async_loop, daemon=True)
        self._thread.start()

    def get_message(self, subject: str):
        if subject not in self._subscriptions or self._subscriptions[subject]._queue.empty():
            return None
        return self._subscriptions[subject]._queue.get_nowait()

    def subscribe(self, subject: str):
        self._control_queue.put(Subscribe(subject=subject))

    def unsubscribe(self, subject: str):
        self._control_queue.put(Unsubscribe(subject=subject))

    def publish(self, subject: str, payload: bytes):
        self._control_queue.put(Publish(subject=subject, payload=payload))

    def shutdown(self):
        self._control_queue.put(Shutdown())

    def _start_async_loop(self):
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(self._run())

    async def _run(self):
        try:
            self._connection = await nats.connect(servers=self._host, error_cb=self._error_handler)
            if self._use_js:
                self._js_context = self._connection.jetstream()
        except ConnectionRefusedError:
            logger.exception(f"Cannot connect to NATS at {self._host}")
            return

        for subject in self._initial_subscriptions:
            await self._subscribe(subject)

        # Main processing loop
        done = False
        while not done:
            if self._control_queue.empty():
                await asyncio.sleep(0.01)
                continue
            el = self._control_queue.get()
            if not hasattr(el, "type"):
                logger.error("Failed to find message type")
                continue

            # Check internal message type
            logger.info(f"Processing message type: {el.type}")
            if el.type == _MessageType.PUBLISH:
                await self._publish(el.subject, el.payload)
            elif el.type == _MessageType.SUBSCRIBE:
                await self._subscribe(el.subject)
            elif el.type == _MessageType.UNSUBSCRIBE:
                await self._unsubscribe(el.subject)
            elif el.type == _MessageType.SHUTDOWN:
                await self._connection.close()
                done = True
            else:
                logger.error(f"Invalid message type sent to NATS queue: {el.type}")

    async def _error_handler(self, error):
        logger.error(f"NATS error: {error}")

    async def _async_sub_handler(self, msg: nats.aio.msg.Msg):
        if msg.subject in self._subscriptions:
            try:
                self._subscriptions[msg.subject]._queue.put(msg.data, block=False)
            except Full:
                logger.warning(f"Message queue full for subject {msg.subject}, dropping message")

    async def _subscribe(self, subject: str, q: Queue | None = None):
        if q is None:
            q = Queue()

        try:
            if self._use_js:
                nats_subscription = await self._js_context.subscribe(
                    subject, cb=self._async_sub_handler
                )
            else:
                nats_subscription = await self._connection.subscribe(
                    subject, cb=self._async_sub_handler
                )
        except TypeError:
            logger.exception(f"Cannot subscribe to {subject} since stream doesn't exist")
            return

        self._subscriptions[subject] = _Subscription(nats_subscription=nats_subscription, queue=q)

    async def _unsubscribe(self, subject: str):
        if subject in self._subscriptions:
            await self._subscriptions[subject]._nats_subscription.unsubscribe()
            del self._subscriptions[subject]
        else:
            logger.error(f"Cannot unsubscribe from {subject} since it's not subscribed")

    async def _publish(self, subject: str, payload: bytes) -> bool:
        try:
            if self._use_js:
                await self._js_context.publish(subject, payload)
            else:
                await self._connection.publish(subject, payload)
        except nats.errors.ConnectionClosedError:
            logger.exception("Failed to send message since NATS connection was closed")
            return False
        except Exception:
            logger.exception(f"Failed to send message to {subject}")
            return False
        else:
            return True
