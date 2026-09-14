/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026, Real-Time Innovations, Inc. All rights reserved.
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

std::map<DDSOperatorBase::ParticipantKey, std::weak_ptr<DDSOperatorBase::ParticipantContext>>
    DDSOperatorBase::participant_contexts_;
std::mutex DDSOperatorBase::participant_contexts_mutex_;

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

  const ParticipantKey participant_key{
      qos_provider_param_.get(), participant_qos_param_.get(), domain_id_param_.get()};
  std::lock_guard<std::mutex> lock(participant_contexts_mutex_);
  auto& cached_context = participant_contexts_[participant_key];
  participant_context_ = cached_context.lock();
  if (!participant_context_) {
    auto qos_provider = dds::core::QosProvider(qos_provider_param_.get());
    auto participant = dds::domain::DomainParticipant(
        domain_id_param_.get(), qos_provider.participant_qos(participant_qos_param_.get()));
    participant_context_ = std::make_shared<ParticipantContext>(
        ParticipantContext{std::move(qos_provider), std::move(participant)});
    cached_context = participant_context_;
  }
  qos_provider_ = participant_context_->qos_provider_;
  participant_ = participant_context_->participant_;
}

}  // namespace holoscan::ops
