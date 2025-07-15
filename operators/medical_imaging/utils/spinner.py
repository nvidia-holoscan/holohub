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

import itertools
import sys
from multiprocessing import Event, Lock, Process


class ProgressSpinner:
    """
    Progress spinner for console.
    """

    def __init__(self, message, delay=0.2):
        self.spinner_symbols = itertools.cycle(["-", "\\", "|", "/"])
        self.delay = delay
        self.stop_event = Event()
        self.spinner_visible = False
        sys.stdout.write(message)

    def __enter__(self):
        self.start()

    def __exit__(self, exception, value, traceback):
        self.stop()

    def _spinner_task(self):
        while not self.stop_event.wait(self.delay):
            self._remove_spinner()
            self._write_next_symbol()

    def _write_next_symbol(self):
        with self._spinner_lock:
            if not self.spinner_visible:
                sys.stdout.write(next(self.spinner_symbols))
                self.spinner_visible = True
                sys.stdout.flush()

    def _remove_spinner(self, cleanup=False):
        with self._spinner_lock:
            if self.spinner_visible:
                sys.stdout.write("\b")
                self.spinner_visible = False
                if cleanup:
                    # overwrite spinner symbol with whitespace
                    sys.stdout.write(" ")
                    sys.stdout.write("\r")
                sys.stdout.flush()

    def start(self):
        """
        Start spinner as a separate process.
        """
        if sys.stdout.isatty():
            self._spinner_lock = Lock()
            self.stop_event.clear()
            self.spinner_process = Process(target=self._spinner_task)
            self.spinner_process.start()

    def stop(self):
        """
        Stop spinner process.
        """
        sys.stdout.write("\b")
        sys.stdout.write("Done")
        if sys.stdout.isatty():
            self.stop_event.set()
            self._remove_spinner(cleanup=True)
            self.spinner_process.join()
            sys.stdout.write("\n")
        else:
            sys.stdout.write("\r")
