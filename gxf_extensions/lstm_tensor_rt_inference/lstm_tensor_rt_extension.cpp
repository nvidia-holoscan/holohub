/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "gxf/std/extension_factory_helper.hpp"

#include "tensor_rt_inference.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xef68668e8e1748f7, 0x88bd37f33565889d, "CustomLSTMInferenceExtension",
                         "TensorRT Custom LSTM Inference extension", "NVIDIA", "0.2.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0x207880b30f3f404a, 0xb3b12659a29f3e26,
                    nvidia::holoscan::lstm_tensor_rt_inference::TensorRtInference,
                    nvidia::gxf::Codelet,
                    "Codelet taking input tensors and feed them into TensorRT for LSTM inference.");
GXF_EXT_FACTORY_END()
