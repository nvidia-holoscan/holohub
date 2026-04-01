/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#ifndef HOLOHUB_HSB_ROCE_RECEIVER_NMD_ROCE_RECEIVER_NO_HOST_METADATA
#define HOLOHUB_HSB_ROCE_RECEIVER_NMD_ROCE_RECEIVER_NO_HOST_METADATA

#include <hololink/operators/roce_receiver/roce_receiver.hpp>

namespace hololink::operators {

/**
 * RoceReceiver variant that keeps metadata on the GPU and avoids host copies.
 */
class RoceReceiverNoHostMetadata final : public RoceReceiver {
 public:
    using RoceReceiver::RoceReceiver;

    void copy_metadata_to_host(unsigned current_page) override;
    const Hololink::FrameMetadata get_frame_metadata() override;
};

}  // namespace hololink::operators

#endif /* HOLOHUB_HSB_ROCE_RECEIVER_NMD_ROCE_RECEIVER_NO_HOST_METADATA */
