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

import logging
from queue import Queue
from typing import Callable, Dict, Optional

from holoscan.core import Tracker

from operators.grpc_operators.python.server.grpc_application import HoloscanGrpcApplication


class ApplicationInstance:
    def __init__(self, instance: HoloscanGrpcApplication, tracker: Optional[Tracker] = None):
        self.instance: HoloscanGrpcApplication = instance
        self.tracker: Optional[Tracker] = tracker
        self.future = None

    def start_application(self):
        self.tracker = Tracker(self.instance)
        self.tracker.__enter__()
        self.future = self.instance.run_async()


class ApplicationFactory(object):
    def __new__(cls):
        """
        Create a singleton instance of the ApplicationFactory class
        """
        if not hasattr(cls, "instance"):
            cls.instance = super(ApplicationFactory, cls).__new__(cls)
            cls.instance.__initialized = False
        return cls.instance

    def __init__(self):
        """
        Initialize the ApplicationFactory class
        """
        if self.__initialized:
            return
        self.applications: Dict[str, Callable[[Queue, Queue], HoloscanGrpcApplication]] = {}
        self.instances: Dict[str, ApplicationInstance] = {}
        self.logger = logging.getLogger(__name__)
        self.__initialized = True

    def register_application(
        self, name: str, application: Callable[[Queue, Queue], HoloscanGrpcApplication]
    ):
        """
        Register an application with the factory
        """

        if name in self.applications:
            self.logger.warning(f"Overwriting existing application: {application}")

        self.applications[name] = application

    def create_new_application_instance(
        self, name: str, request_queue: Queue, response_queue: Queue
    ):
        if name not in self.applications:
            self.logger.error(f"Application {name} not found in the registry.")
            return None

        if name in self.instances:
            self.logger.warning(f"Another application instance is running: {name}")

        self.logger.info(f"Creating new application instance {name}")

        func = self.applications[name]
        self.instances[name] = func(request_queue, response_queue)
        return self.instances[name].instance

    def destroy_application_instance(self, application_instance: HoloscanGrpcApplication):
        """
        Destroy an application instance
        """
        for service_name, instance in self.instances.items():
            if instance.instance == application_instance:
                instance.tracker.tracker.print()
                instance.tracker.__exit__(None, None, None)
                self.logger.info(f"Application instance deleted for {service_name}")
                del self.instances[service_name]
                break
