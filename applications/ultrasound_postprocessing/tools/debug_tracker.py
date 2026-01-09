# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

try:
    import holoscan

    print(f"Holoscan version: {holoscan.__version__}")
except ImportError:
    print("Holoscan not installed.")
    sys.exit(1)

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator

# Try to import Tracker
try:
    from holoscan.core import Tracker

    print("Imported Tracker from holoscan.core")
except ImportError:
    try:
        from holoscan.tracking import Tracker

        print("Imported Tracker from holoscan.tracking")
    except ImportError:
        print("Could not import Tracker.")
        sys.exit(1)


class PingOp(Operator):
    def setup(self, spec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        print("PingOp executing...")
        op_output.emit(1, "out")


class SinkOp(Operator):
    def setup(self, spec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        print(f"Sink received: {msg}")


class SimpleApp(Application):
    def compose(self):
        # Run 5 times
        ping = PingOp(self, CountCondition(self, 5), name="ping")
        sink = SinkOp(self, name="sink")
        self.add_flow(ping, sink)


def main():
    log_file = "debug_tracker.log"
    if os.path.exists(log_file):
        os.remove(log_file)

    app = SimpleApp()
    print(f"Running minimal app with Tracker -> {log_file}")

    try:
        with Tracker(app, filename=log_file) as tracker:  # noqa: F841
            app.run()
    except Exception as e:
        print(f"Error: {e}")

    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"Success! Log file created. Size: {size} bytes")
    else:
        print("Failure! Log file was NOT created.")


if __name__ == "__main__":
    main()
