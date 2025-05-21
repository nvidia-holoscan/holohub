# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Import the holoscan modules we'll depend on
import holoscan.core
import holoscan.gxf

# Load the python binding
try:
    from ._@MODULE_NAME@ import @MODULE_CLASS_NAME@
except ImportError as e:
    pybind11_hsdk_err = 'unknown base type "holoscan::'

    if not pybind11_hsdk_err in str(e):
        # Unknown import error, raise it
        raise e

    # Provide information regarding pybind11 ABI protection
    note = """
- Holoscan SDK >= 3.3.0: make sure to link your bindings against 'holoscan::pybind11'.
- Holoscan SDK < 3.3.0: use the same compiler version as your installation of the Holoscan SDK.

See https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator_python_bindings.html#pybind11-abi-compatibility for details.
"""

    # Raise with note if available (Python 3.11+) ...
    if hasattr(e, "add_note"):
        e.add_note(note)
        raise e

    # ... or raise new exception with same trace and message
    e = ImportError(e.msg + "\n" + note).with_traceback(e.__traceback__)
    raise e from None


# Register types with the SDK
try:
    # If a register_types function exists, register the types with the SDK
    from ._@MODULE_NAME@ import register_types as _register_types

    try:
        from holoscan.core import io_type_registry
    except ImportError as e:
        import warnings
        warnings.warn(
            "`holoscan.core.io_type_registry` is unavailable in Holoscan SDK < 2.1.0. "
            "To use a user-defined `register_types` function, you must upgrade Holoscan SDK."
        )
        raise e

    # register any custom emitter/receiver types with the SDK's registry
    _register_types(io_type_registry)
except ImportError as e:
    # Most extensions will not provide a user-defined `register_types` function, so don't warn or
    # raise an error in that case.
    pass
