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

/**
 * Holoscan Slang Attribute Definitions
 *
 * This file defines custom attributes used by the Holoscan framework to annotate
 * Slang shader code for GPU compute operations. These attributes help the Holoscan
 * runtime understand how to bind resources and configure compute kernels.
 *
 * Resource Name String Construction
 *
 * Resource names in Holoscan attributes follow a specific format to identify and bind
 * resources to shader variables. The name string can take several forms:
 *
 * 1. Simple Resource Name:
 *    - Format: "resource_name"
 *    - Example: "input_buffer", "output_tensor"
 *    - Used when referencing a single resource directly
 *
 * 2. Tensor Map Reference:
 *    - Format: "tensor_map_name:tensor_name"
 *    - Example: "model:weights", "data:input_image"
 *    - Used when the resource is part of a named tensor map, where the part before
 *      the colon identifies the tensor map and the part after identifies the specific
 *      tensor within that map
 *
 * 3. \anchor swizzle Resource with swizzle (for allocation and invocations size attributes):
 *    - Format: "resource_name.swizzle_string"
 *    - Example: "input_tensor:image.cx", "output_buffer.xy"
 *    - The swizzle string selects specific dimensions of the resource for size matching
 *    - Allowed characters: "x", "y", "z", "c", "0" - "9"
 *      where "c" is the component count and "0" - "9" static values.
 *
 * Examples:
 * - [holoscan_input("input_data")] - Binds to a resource named "input_data"
 * - [holoscan_output("model:output")] - Binds to the "output" tensor in the "model" tensor map
 * - [holoscan_alloc_size_of("input_tensor.cx")] - Allocates based on the x dimension of
 * "input_tensor"
 */

/**
 * Marks a variable as an input resource for a compute kernel.
 * Used to specify buffers, textures, or other data that the kernel reads from.
 *
 * @param name The identifier for the input resource, used by the Holoscan runtime
 *             to bind the appropriate data source.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_inputAttribute {
  string name;
};

/**
 * Marks a variable as an output resource for a compute kernel.
 * Used to specify buffers, textures, or other data that the kernel writes to.
 *
 * @param name The identifier for the output resource, used by the Holoscan runtime
 *             to bind the appropriate data destination.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_outputAttribute {
  string name;
};

/**
 * Marks a variable as a parameter that can be configured at runtime.
 * Used for values that can be modified without recompiling the shader.
 *
 * @param name The identifier for the parameter, used by the Holoscan runtime
 *             to set the parameter value dynamically. The name can be followed by an equals sign
 *             and a default value. The default value must be a string which can be converted to
 *             the type of the parameter. If the default value is not provided, the parameter is
 *             initialized to the default value of the parameter type.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_parameterAttribute {
  string name;
};

/**
 * Set the value of a parameter to the size of a resource. The parameter must be a uint3.
 *
 * @param name The name of the reference resource whose size will be set to the parameter.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_size_ofAttribute {
  string name;
};

/**
 * Set the value of a parameter to the strides of a resource. The parameter must be a uint64_t3. The
 * strides are the number of bytes between elements in each dimension. E.g. the x component of the
 * strides is the number of bytes between elements in the x dimension.
 *
 * @param name The name of the reference resource whose strides will be set to the parameter.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_strides_ofAttribute {
  string name;
};

/**
 * Specifies the allocation size of a resource based on another resource's dimensions.
 * Used when allocating memory for a resource that needs to match another resource's size.
 * The storage type of the allocated resource is set to the storage type of the reference resource.
 *
 * @param name The name of the reference resource whose allocation size should be matched. Supports
 * a swizzle string after a dot. See \ref swizzle "Resource with swizzle" section above for format
 * details.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_alloc_size_ofAttribute {
  string name;
};

/**
 * Specifies the allocation size of a resource. The storage type is set to device.
 *
 * @param x Number of elements in the X dimension
 * @param y Number of elements in the Y dimension
 * @param z Number of elements in the Z dimension
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_allocAttribute {
  uint x;
  uint y;
  uint z;
};

/**
 * Specifies the invocation size for a compute kernel.
 * Defines how many invocations will be launched in each dimension.
 *
 * @param x Number of invocations in the X dimension
 * @param y Number of invocations in the Y dimension
 * @param z Number of invocations in the Z dimension
 */
[__AttributeUsage(_AttributeTargets.Function)]
public struct holoscan_invocationsAttribute {
  uint x;
  uint y;
  uint z;
};

/**
 * Specifies the invocation size for a compute kernel based on a resource's size.
 * The invocation size is automatically calculated based on the dimensions of the specified
 * resource.
 *
 * @param name The name of the resource whose size should be used to calculate invocation size.
 *             Supports swizzle string after a dot. See \ref swizzle "Resource with swizzle" section
 * above for format details.
 */
[__AttributeUsage(_AttributeTargets.Function)]
public struct holoscan_invocations_size_ofAttribute {
  string name;
};

/**
 * Specifies that the resource should be initialized to zero.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_zerosAttribute {};
