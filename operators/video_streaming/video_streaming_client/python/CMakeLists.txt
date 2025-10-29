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

include(pybind11_add_holohub_module)

# Use the actual C++ target name (video_streaming_client) to get proper include directories
pybind11_add_holohub_module(
    CPP_CMAKE_TARGET video_streaming_client
    CLASS_NAME "VideoStreamingClientOp"
    SOURCES video_streaming_client.cpp
)

# Override the output name and directory to match expected module structure
# This ensures the .so file is named _video_streaming_client to match PYBIND11_MODULE
set_target_properties(video_streaming_client_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HOLOHUB_PYTHON_MODULE_OUT_DIR}/video_streaming_client"
    OUTPUT_NAME "_video_streaming_client"
)

# Use file(GENERATE) to create __init__.py at generation time
# Note: file(GENERATE) will overwrite any existing __init__.py, so no manual removal needed
set(CLIENT_INIT_PY_CONTENT "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import holoscan.core
import holoscan.gxf

# Load the python binding
try:
    from ._video_streaming_client import StreamingClientOp
except ImportError as e:
    pybind11_hsdk_err = 'unknown base type \"holoscan::'
    if pybind11_hsdk_err not in str(e):
        raise e
    note = '''
- Holoscan SDK >= 3.3.0: make sure to link your bindings against 'holoscan::pybind11'.
- Holoscan SDK < 3.3.0: use the same compiler version as your installation of the Holoscan SDK.

See https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator_python_bindings.html#pybind11-abi-compatibility for details.
'''
    if hasattr(e, 'add_note'):
        e.add_note(note)
        raise e
    e = ImportError(str(e) + '\\n' + note).with_traceback(e.__traceback__)
    raise e from None

# Register types with the SDK
try:
    from ._video_streaming_client import register_types as _register_types
    try:
        from holoscan.core import io_type_registry
    except ImportError as e:
        import warnings
        warnings.warn(
            '`holoscan.core.io_type_registry` is unavailable in Holoscan SDK < 2.1.0. '
            'To use a user-defined `register_types` function, you must upgrade Holoscan SDK.'
        )
        raise e
    _register_types(io_type_registry)
except ImportError as e:
    pass
")

file(GENERATE
    OUTPUT "${HOLOHUB_PYTHON_MODULE_OUT_DIR}/video_streaming_client/__init__.py"
    CONTENT "${CLIENT_INIT_PY_CONTENT}"
)
