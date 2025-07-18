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
    List of regex patterns to apply to message unique IDs. If empty, all messages not matching a
    denylist pattern will be logged. Otherwise, there must be a match to one of the allowlist
    patterns.
denylist_patterns : list of str, optional
    List of regex patterns to apply to message unique IDs. If specified and there is a match at
    least one of these patterns, the message is not logged.
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

The pattern matching logic is as follows:

  - If `denylist patterns` is specified and there is a match, do not log it.
  - Next check if `allowlist_patterns` is empty:
    - If yes, return true (allow everything)
    - If no, return true only if there is a match to at least one of the specified patterns.
)doc")
}  // namespace holoscan::doc::MessageLogger
