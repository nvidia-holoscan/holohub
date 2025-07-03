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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for vector

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

#include "./message_logger_pydoc.hpp"

#include "../message_logger.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resources/data_logger.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::data_loggers {

class PyMessageLogger : public MessageLogger {
 public:
  /* Inherit the constructors */
  using MessageLogger::MessageLogger;

  // Define a constructor that fully initializes the object.
  PyMessageLogger(Fragment* fragment, const py::args& args,
                       bool log_inputs = true, bool log_outputs = true,
                       bool log_tensor_data_content = false, bool log_metadata = false,
                       const std::vector<std::string>& allowlist_patterns = {},
                       const std::vector<std::string>& denylist_patterns = {},
                       const std::string& name = "message_logger")
      : MessageLogger(ArgList{Arg{"log_inputs", log_inputs},
                                   Arg{"log_outputs", log_outputs},
                                   Arg{"log_tensor_data_content", log_tensor_data_content},
                                   Arg{"log_metadata", log_metadata},
                                   Arg{"allowlist_patterns", allowlist_patterns},
                                   Arg{"denylist_patterns", denylist_patterns}}) {
    // warn if non-empty allowlist and denylist are specified
    if (!allowlist_patterns.empty() && !denylist_patterns.empty()) {
      std::string warning_msg =
          "MessageLogger: Both allowlist_patterns and denylist_patterns are specified. "
          "Allowlist takes precedence and denylist will be ignored.";
      try {
        auto warnings = py::module_::import("warnings");
        warnings.attr("warn")(
            warning_msg, py::arg("category") = py::module_::import("builtins").attr("UserWarning"));
      } catch (const py::error_already_set&) {
        // If we can't import warnings or we're not in a Python context, just continue
      } catch (...) {
        // Ignore any other Python-related errors
      }
    }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }

  void initialize() override {
    HOLOSCAN_LOG_TRACE("MessageLogger::initialize");

    // call parent initialize after adding missing serializer arg above
    MessageLogger::initialize();
  }
};

/* The python module */

PYBIND11_MODULE(_message_logger, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK MessageLogger Python Bindings
        -----------------------------------------------
        .. currentmodule:: _message_logger
    )pbdoc";

  py::class_<MessageLogger,
             PyMessageLogger,
             DataLoggerResource,
             std::shared_ptr<MessageLogger>>(
      m, "MessageLogger", doc::MessageLogger::doc_MessageLogger)
      .def(py::init<Fragment*,
                    const py::args&,
                    bool,
                    bool,
                    bool,
                    bool,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::string&>(),
           "fragment"_a,
           "log_inputs"_a = true,
           "log_outputs"_a = true,
           "log_tensor_data_content"_a = false,
           "log_metadata"_a = false,
           "allowlist_patterns"_a = py::list(),
           "denylist_patterns"_a = py::list(),
           "name"_a = "message_logger"s,
           doc::MessageLogger::doc_MessageLogger);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::data_loggers
