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

#include <holoscan/holoscan.hpp>
#include <string>

namespace holoscan::apps {

/**
 * @brief Subscriber application that receives and displays rendered frames via Holoviz
 *
 * This application:
 * - Connects to the publisher via UCXX
 * - Receives pre-rendered frames with tool tracking overlays
 * - Displays the frames with HolovizOp (simple image display)
 * - Optionally records received frames for validation (if recorder config exists in YAML)
 */
class UcxxEndoscopySubscriberHolovizApp : public holoscan::Application {
 public:
  void set_hostname(const std::string& hostname) { hostname_ = hostname; }
  void set_port(int port) { port_ = port; }

  void compose() override;

 private:
  std::string hostname_ = "127.0.0.1";
  int port_ = 50008;
};

}  // namespace holoscan::apps


