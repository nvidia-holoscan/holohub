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

from queue import Queue
from threading import Condition

from holoscan.core import Resource


class ConditionVariableQueue(Resource):
    def __init__(self, fragment, queue: Queue = Queue(), *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.condition = Condition()
        self.queue = queue

    def push(self, value):
        with self.condition:
            self.queue.put(value)
            self.condition.notify()

    def pop(self):
        with self.condition:
            while self.queue.empty():
                self.condition.wait()
        if self.empty():
            return None
        item = self.queue.get()
        return item

    def empty(self):
        return self.queue.empty()

    def iterator(self):
        while True:
            yield self.pop()
