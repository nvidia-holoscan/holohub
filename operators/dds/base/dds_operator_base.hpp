/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dds/core/ddscore.hpp>

namespace holoscan::ops {

/**
 * @brief Base class for a DDS operator.
 */
class DDSOperatorBase : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DDSOperatorBase)

  DDSOperatorBase() = delete;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

 protected:
  dds::core::QosProvider qos_provider_ = dds::core::null;
  dds::domain::DomainParticipant participant_ = dds::core::null;

 private:
  Parameter<std::string> qos_provider_param_;
  Parameter<std::string> participant_qos_param_;
  Parameter<uint32_t> domain_id_param_;

  struct DomainParticipantEntry {
    explicit DomainParticipantEntry(dds::core::QosProvider qos_provider,
                                    std::string participant_qos,
                                    uint32_t domain_id,
                                    dds::domain::DomainParticipant participant)
        : qos_provider_(qos_provider),
          participant_qos_(participant_qos),
          domain_id_(domain_id),
          participant_(participant)
    {}

    dds::core::QosProvider qos_provider_;
    std::string participant_qos_;
    uint32_t domain_id_;
    dds::domain::DomainParticipant participant_;
  };

  static std::map<std::string, dds::core::QosProvider> qos_providers_;
  static std::vector<DomainParticipantEntry> participants_;
};

}  // namespace holoscan::ops
