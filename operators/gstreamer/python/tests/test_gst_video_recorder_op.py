# SPDX-FileCopyrightText: Copyright (c) 2026, TECNALIA. All rights reserved.
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

from holoscan.core import Application

from holohub.holoscan_gstreamer_bridge import GstVideoRecorderOp


class _TestApplication(Application):
    def compose(self):
        pass


def test_import_gst_video_recorder_op():
    assert GstVideoRecorderOp.__name__ == "GstVideoRecorderOp"


def test_construct_with_defaults(tmp_path):
    app = _TestApplication()
    op = GstVideoRecorderOp(
        app,
        filename=str(tmp_path / "output.mp4"),
    )
    assert op is not None


def test_construct_with_kwargs(tmp_path):
    app = _TestApplication()
    op = GstVideoRecorderOp(
        app,
        filename=str(tmp_path / "output.mp4"),
        max_buffers=4,
        properties={"bitrate": "8000"},
        name="recorder",
    )
    assert op is not None
