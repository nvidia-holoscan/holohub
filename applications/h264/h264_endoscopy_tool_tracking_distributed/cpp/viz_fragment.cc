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

#ifndef VIZ_FRAGMENT_CC
#define VIZ_FRAGMENT_CC

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

class VizFragment : public holoscan::Fragment {
 private:
  uint32_t width_ = 0;
  uint32_t height_ = 0;

 public:
  VizFragment(const uint32_t width, const uint32_t height) : width_(width), height_(height) {}

  void compose() override {
    std::shared_ptr<UnboundedAllocator> visualizer_allocator;

    auto visualizer_operator =
        make_operator<ops::HolovizOp>("holoviz",
                                      from_config("holoviz"),
                                      Arg("width") = width_,
                                      Arg("height") = height_,
                                      Arg("allocator") = visualizer_allocator);
    add_operator(visualizer_operator);
  }
};

#endif