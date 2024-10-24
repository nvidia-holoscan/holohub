# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import time

import zmq


class MessageSender:
    def __init__(self, endpoint: str = "tcp://*:5556", add_sleep: bool = True):
        self.endpoint_ = endpoint
        self.context_ = zmq.Context.instance()
        self.publisher_ = self.context_.socket(zmq.PUB)
        self.publisher_.bind(self.endpoint_)
        # Sleep to allow the subscribers to connect to the publisher, see 'slow joiner'
        # symptom https://zguide.zeromq.org/docs/chapter1/#Getting-the-Message-Out

        # This is only needed when the application logic is such that the Sender is
        # created and the send function is called right away, e.g. when the caller
        # uses lazy init of the sender when message is ready to be sent.
        # Cases where caller inits the sender, and will have a nature delay in the app
        # before the first message is ready, the sleep is not needed.
        if add_sleep:
            time.sleep(0.1)

    def stop(self):
        self.publisher_.disconnect(self.endpoint_)
        self.context_.destroy()

    def send_json(self, topic: str, data: json):
        self.publisher_.send_multipart([topic.encode("utf-8"), json.dumps(data).encode("utf-8")])


class MessageReceiver:
    def __init__(self, topic: str, endpoint: str = "tcp://localhost:5556"):
        self.topic_ = topic
        self.endpoint_ = endpoint
        self.context_ = zmq.Context.instance()
        self.subscriber_ = self.context_.socket(zmq.SUB)
        self.subscriber_.subscribe(self.topic_)
        self.subscriber_.connect(self.endpoint_)

    def stop(self):
        self.subscriber_.disconnect(self.endpoint_)
        self.context_.destroy()

    def receive_json(self, blocking: bool = False):
        as_json = dict()
        try:
            if blocking:
                topic, data = self.subscriber_.recv_multipart()
            else:
                topic, data = self.subscriber_.recv_multipart(zmq.NOBLOCK)
        except zmq.ZMQError as e:
            if e.errno != zmq.Errno.EAGAIN:
                print(f"ZeroMQ error {e}")
        else:
            as_json = json.loads(data.decode("utf-8"))

        return as_json
