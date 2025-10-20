/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef COMMON_ASYNCHRONOUS_CONDITION_QUEUE_HPP
#define COMMON_ASYNCHRONOUS_CONDITION_QUEUE_HPP

#include <memory>
#include <queue>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

using namespace holoscan;

/**
 * @class AsynchronousConditionQueue
 * @brief This class is a Holoscan Resource that is responsible for storing a queue of data
 * entities.
 *
 * The AsynchronousCondition is used to notify when data is available.
 */

template <typename DataT>
class AsynchronousConditionQueue : public Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(AsynchronousConditionQueue, Resource)

  explicit AsynchronousConditionQueue(
      std::shared_ptr<AsynchronousCondition> request_available_condition)
      : data_available_condition_(request_available_condition), queue_() {}

  void push(DataT entity) {
    queue_.push(entity);
    if (data_available_condition_->event_state() == AsynchronousEventState::EVENT_WAITING) {
      data_available_condition_->event_state(AsynchronousEventState::EVENT_DONE);
    }
  }

  DataT pop() {
    if (empty()) { return nullptr; }
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  bool empty() { return queue_.empty(); }

 private:
  std::shared_ptr<AsynchronousCondition> data_available_condition_;
  std::queue<DataT> queue_;
};

}  // namespace holoscan::ops

#endif /* COMMON_ASYNCHRONOUS_CONDITION_QUEUE_HPP */
