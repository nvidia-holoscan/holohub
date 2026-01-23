/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "pydoc.hpp"

#include "holoscan/core/arg.hpp"
#include "holoscan/python/core/component_util.hpp"
#include "nats_logger.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::data_loggers {

class PyNatsLogger : public NatsLogger {
 public:
  /* Inherit the constructors */
  using NatsLogger::NatsLogger;

  // Define a constructor that fully initializes the object.
  PyNatsLogger(Fragment* fragment, const std::string& nats_url = "nats://0.0.0.0:4222",
               const std::string& subject_prefix = "", float publish_rate = 1.0,
               bool log_inputs = true, bool log_outputs = true, bool log_metadata = true,
               bool log_tensor_data_content = true, bool use_scheduler_clock = false,
               std::optional<std::shared_ptr<Resource>> clock = std::nullopt,
               const std::vector<std::string>& allowlist_patterns = {},
               const std::vector<std::string>& denylist_patterns = {},
               size_t max_queue_size = 50000, int64_t worker_sleep_time = 50000,
               // when loading the enum from the YAML config via **app.kwargs(key) it will become a
               // string. use of std::variant here is to support that case
               std::variant<AsyncQueuePolicy, std::string> queue_policy = AsyncQueuePolicy::kReject,
               size_t large_data_max_queue_size = 1000,
               int64_t large_data_worker_sleep_time = 50000,
               std::variant<AsyncQueuePolicy, std::string> large_data_queue_policy =
                   AsyncQueuePolicy::kReject,
               bool enable_large_data_queue = true, const std::string& name = "nats_logger")
      : NatsLogger(ArgList{Arg{"nats_url", nats_url},
                           Arg{"subject_prefix", subject_prefix},
                           Arg{"publish_rate", publish_rate},
                           Arg{"log_inputs", log_inputs},
                           Arg{"log_outputs", log_outputs},
                           Arg{"log_metadata", log_metadata},
                           Arg{"log_tensor_data_content", log_tensor_data_content},
                           Arg{"use_scheduler_clock", use_scheduler_clock},
                           Arg{"allowlist_patterns", allowlist_patterns},
                           Arg{"denylist_patterns", denylist_patterns},
                           Arg{"max_queue_size", max_queue_size},
                           Arg{"worker_sleep_time", worker_sleep_time},
                           Arg{"large_data_max_queue_size", large_data_max_queue_size},
                           Arg{"large_data_worker_sleep_time", large_data_worker_sleep_time},
                           Arg{"enable_large_data_queue", enable_large_data_queue}}) {
    if (clock.has_value()) {
      this->add_arg(Arg{"clock", clock.value()});
    }
    if (std::holds_alternative<std::string>(queue_policy)) {
      // C++ layer supports YAML::Node -> enum conversion via the registered argument setter
      this->add_arg(Arg("queue_policy", YAML::Node(std::get<std::string>(queue_policy))));
    } else {
      this->add_arg(Arg("queue_policy", std::get<holoscan::AsyncQueuePolicy>(queue_policy)));
    }
    if (std::holds_alternative<std::string>(large_data_queue_policy)) {
      // C++ layer supports YAML::Node -> enum conversion via the registered argument setter
      this->add_arg(Arg("large_data_queue_policy",
                        YAML::Node(std::get<std::string>(large_data_queue_policy))));
    } else {
      this->add_arg(Arg("large_data_queue_policy",
                        std::get<holoscan::AsyncQueuePolicy>(large_data_queue_policy)));
    }
    init_component_base(this, fragment, name);
  }
};
/* The python module */

PYBIND11_MODULE(_nats_logger, m) {
  m.doc() = R"pbdoc(
         Holoscan SDK NatsLogger Python Bindings
         -----------------------------------------------
         .. currentmodule:: _nats_logger
     )pbdoc";

  py::class_<NatsLogger,
             PyNatsLogger,
             AsyncDataLoggerResource,
             DataLoggerResource,
             std::shared_ptr<NatsLogger>>(m, "NatsLogger", doc::NatsLogger::doc_NatsLogger)
      .def(py::init<Fragment*,
                    const std::string&,
                    const std::string&,
                    float,
                    bool,
                    bool,
                    bool,
                    bool,
                    bool,
                    std::optional<std::shared_ptr<Resource>>,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    size_t,
                    int64_t,
                    std::variant<AsyncQueuePolicy, std::string>,
                    size_t,
                    int64_t,
                    std::variant<AsyncQueuePolicy, std::string>,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "nats_url"_a = "nats://0.0.0.0:4222"s,
           "subject_prefix"_a = ""s,
           "publish_rate"_a = 1.0,
           "log_inputs"_a = true,
           "log_outputs"_a = true,
           "log_metadata"_a = true,
           "log_tensor_data_content"_a = true,
           "use_scheduler_clock"_a = false,
           "clock"_a = py::none(),
           "allowlist_patterns"_a = py::list(),
           "denylist_patterns"_a = py::list(),
           "max_queue_size"_a = 50000,
           "worker_sleep_time"_a = 50000,
           "queue_policy"_a = AsyncQueuePolicy::kReject,
           "large_data_max_queue_size"_a = 1000,
           "large_data_worker_sleep_time"_a = 50000,
           "large_data_queue_policy"_a = AsyncQueuePolicy::kReject,
           "enable_large_data_queue"_a = true,
           "name"_a = "nats_logger"s,
           doc::NatsLogger::doc_NatsLogger);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::data_loggers
