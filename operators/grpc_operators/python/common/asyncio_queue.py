# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Resource


class AsyncIoQueue(Resource):
    def __init__(self, fragment, queue: asyncio.Queue = None, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.queue = queue if queue is not None else asyncio.Queue()

    async def push(self, value):
        await self.queue.put(value)

    async def pop(self):
        return await self.queue.get()

    def empty(self):
        return self.queue.empty()

    async def iterator(self):
        while True:
            yield await self.pop()
