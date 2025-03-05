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

from holoscan.conditions import AsynchronousCondition, AsynchronousEventState
from holoscan.core import Fragment, Resource


class AsynchronousConditionQueue(Resource):
    def __init__(self, fragment: Fragment, condition: AsynchronousCondition, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.condition = condition
        self.queue = Queue()

    def push(self, value):
        self.queue.put(value)
        if self.condition.event_state == AsynchronousEventState.EVENT_WAITING:
            self.condition.event_state = AsynchronousEventState.EVENT_DONE

    def pop(self):
        if not self.queue.empty():
            return self.queue.get()
        else:
            return None

    def empty(self):
        return self.queue.empty()
