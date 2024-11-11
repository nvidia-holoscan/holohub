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

#ifndef COMMON_CONDITIONAL_VARIABLE_QUEUE_HPP
#define COMMON_CONDITIONAL_VARIABLE_QUEUE_HPP

#include <queue>

#include <holoscan/holoscan.hpp>

#include "holoscan.pb.h"

using holoscan::entity::EntityResponse;

namespace holoscan::ops {

using namespace holoscan;

/*
 * @class ConditionVariableQueue
 * @brief This class is a Holoscan Resource that is responsible for storing a queue of data entities.
 *
 * A mutex and a condition variable are used to control access to the queue.
 */
template <typename DataT>
class ConditionVariableQueue : public Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ConditionVariableQueue, Resource)
  ConditionVariableQueue() : queue_() {}
  explicit ConditionVariableQueue(std::queue<DataT>& queue) : queue_(queue) {}

  void push(DataT value) {
    std::lock_guard<std::mutex> lock(response_available_mutex_);
    queue_.push(value);
    data_available_condition_.notify_one();
  }

  DataT pop() {
    std::unique_lock<std::mutex> lock(response_available_mutex_);
    data_available_condition_.wait(lock, [this]() { return !queue_.empty(); });
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  bool empty() {
    return queue_.empty();
  }

 private:
  std::queue<DataT> queue_;
  std::condition_variable data_available_condition_;
  std::mutex response_available_mutex_;
};

}  // namespace holoscan::ops

#endif /* COMMON_CONDITIONAL_VARIABLE_QUEUE_HPP */
