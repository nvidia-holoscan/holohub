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

import os
from holoscan.core import Application

class BenchmarkedApplication(Application):
    def run(self):
        print ("Running benchmarked application")
        tracker = self.track()

        # Get the data flow tracking log file from environment variable
        flow_tracking_log_file = os.environ.get("HOLOSCAN_DATA_FLOW_LOG_FILE", None)
        if flow_tracking_log_file:
            tracker.enable_logging(flow_tracking_log_file)
        else:
            tracker.enable_logging()

        # Load scheduler parameters from environment variables
        super().run()