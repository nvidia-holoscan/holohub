/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc::MessageLogger {

PYDOC(MessageLogger, R"doc(
Basic message logger for debugging and development purposes.

This data logger is a basic example that only indicates when a message was emitted or received.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the data logger belongs to.
log_inputs : bool, optional
    Whether to log input messages. Default is True.
log_outputs : bool, optional
    Whether to log output messages. Default is True.
log_tensor_data_content : bool, optional
    Whether to log the actual content of tensor data. Default is False.
log_metadata : bool, optional
    Whether to log metadata associated with messages. Default is True.
allowlist_patterns : list of str, optional
    List of regex patterns. Only messages matching these patterns will be logged.
    If empty, all messages are allowed.
denylist_patterns : list of str, optional
    List of regex patterns. Messages matching these patterns will be filtered out.
    If any `allowlist_patterns` are specified, those take precendence and
    `denylist_patterns` is not used.
name : str, optional (constructor only)
    The name of the data logger. Default value is ``"message_logger"``.

Notes
-----
If `allowlist_patterns` or `denylist_patterns` are specified, they are applied to the `unique_id`
assigned to messages by the underlying framework.

In a non-distributed application (without a fragment name), the unique_id for a message will have
one of the following forms:

- operator_name.port_name
- operator_name.port_name:index  (for multi-receivers with N:1 connection)

For distributed applications, the fragment name will also appear in the unique id:

- fragment_name.operator_name.port_name
- fragment_name.operator_name.port_name:index  (for multi-receivers with N:1 connection)
)doc")
}  // namespace holoscan::doc::MessageLogger
