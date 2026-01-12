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

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET video_streaming_server
    CLASS_NAME "StreamingServerResource;StreamingServerUpstreamOp;StreamingServerDownstreamOp"
    SOURCES video_streaming_server.cpp
)

# Custom __init__.py with all three classes explicitly listed
set(INIT_PY_CONTENT "# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import holoscan.core
import holoscan.gxf

# Load the python binding
try:
    from ._video_streaming_server import (
        StreamingServerResource,
        StreamingServerUpstreamOp,
        StreamingServerDownstreamOp
    )
except ImportError as e:
    pybind11_hsdk_err = 'unknown base type \"holoscan::'
    if pybind11_hsdk_err not in str(e):
        raise e
    note = \"\"\"
- Holoscan SDK >= 3.3.0: make sure to link your bindings against 'holoscan::pybind11'.
- Holoscan SDK < 3.3.0: use the same compiler version as your installation of the Holoscan SDK.

See https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator_python_bindings.html#pybind11-abi-compatibility for details.
\"\"\"
    if hasattr(e, \"add_note\"):
        e.add_note(note)
        raise e
    e = ImportError(str(e) + \"\\n\" + note).with_traceback(e.__traceback__)
    raise e from None

# Register types with the SDK
try:
    from ._video_streaming_server import register_types as _register_types
    try:
        from holoscan.core import io_type_registry
    except ImportError as e:
        import warnings
        warnings.warn(
            \"`holoscan.core.io_type_registry` is unavailable in Holoscan SDK < 2.1.0. \"
            \"To use a user-defined `register_types` function, you must upgrade Holoscan SDK.\"
        )
        raise e
    _register_types(io_type_registry)
except ImportError as e:
    pass
")

# Write the custom __init__.py at generation time (after configure, before build)
# Use file(GENERATE) which runs after all configure-time processing
file(GENERATE
    OUTPUT "${HOLOHUB_PYTHON_MODULE_OUT_DIR}/video_streaming_server/__init__.py"
    CONTENT "${INIT_PY_CONTENT}"
)

# Add Python binding tests if testing is enabled and Python bindings are built
if(BUILD_TESTING AND HOLOHUB_BUILD_PYTHON)
    include(add_python_tests)
    add_python_tests(
        INPUT "tests"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PREFIX "video_streaming_server_pytest"
        PYTEST_ARGS "-v"
    )
endif()
