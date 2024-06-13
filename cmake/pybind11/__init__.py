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

import warnings

import holoscan.core
import holoscan.gxf

from ._@MODULE_NAME@ import @MODULE_CLASS_NAME@

try:
    # If a register_types function exists, register the types with the SDK
    from ._@MODULE_NAME@ import register_types as _register_types

    try:
        from holoscan.core import io_type_registry
    except ImportError as e:
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
