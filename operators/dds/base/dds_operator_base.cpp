/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dds_operator_base.hpp"

namespace holoscan::ops {

std::map<std::string, dds::core::QosProvider> DDSOperatorBase::qos_providers_;
std::vector<DDSOperatorBase::DomainParticipantEntry> DDSOperatorBase::participants_;

void DDSOperatorBase::setup(OperatorSpec& spec) {
  spec.param(qos_provider_param_, "qos_provider", "QoS Provider",
             "URI for the QosProvider", std::string("qos_profiles.xml"));
  spec.param(participant_qos_param_, "participant_qos", "Participant QoS",
             "Domain Participant QoS Profile", std::string());
  spec.param(domain_id_param_, "domain_id", "Domain ID",
             "Domain Participant ID", 0u);
}

void DDSOperatorBase::initialize() {
  Operator::initialize();

  // Find (or create) the QoSProvider.
  auto qos_provider_it = qos_providers_.find(qos_provider_param_.get());
  if (qos_provider_it == qos_providers_.end()) {
    qos_provider_it = qos_providers_.insert(qos_providers_.end(),
        std::pair{qos_provider_param_.get(), dds::core::QosProvider(qos_provider_param_.get())});
  }
  qos_provider_ = qos_provider_it->second;

  // Find (or create) the DomainParticipant.
  for (const auto& entry : participants_) {
    if (entry.qos_provider_ == qos_provider_ &&
        entry.participant_qos_ == participant_qos_param_.get() &&
        entry.domain_id_ == domain_id_param_.get()) {
      participant_ = entry.participant_;
      break;
    }
  }
  if (participant_ == dds::core::null) {
    participant_ = dds::domain::DomainParticipant(
        domain_id_param_.get(), qos_provider_.participant_qos(participant_qos_param_.get()));
    participants_.push_back(DomainParticipantEntry(
        qos_provider_, participant_qos_param_.get(), domain_id_param_.get(), participant_));
  }
}

}  // namespace holoscan::ops
