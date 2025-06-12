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
 */

/**
 * Marks a variable as an input resource for a compute kernel.
 * Used to specify buffers, textures, or other data that the kernel reads from.
 *
 * @param name The identifier for the input resource, used by the Holoscan runtime
 *             to bind the appropriate data source.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_inputAttribute
{
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
public struct holoscan_outputAttribute
{
    string name;
};

/**
 * Marks a variable as a parameter that can be configured at runtime.
 * Used for values that can be modified without recompiling the shader.
 *
 * @param name The identifier for the parameter, used by the Holoscan runtime
 *             to set the parameter value dynamically.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_parameterAttribute
{
    string name;
};

/**
 * Set the value of a parameter to the size of a resource.
 *
 * @param name The name of the reference resource whose size will be set to the parameter.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_size_ofAttribute
{
    string name;
};

/**
 * Specifies the allocation size of a resource based on another resource's dimensions.
 * Used when allocating memory for a resource that needs to match another resource's size.
 *
 * @param name The name of the reference resource whose allocation size should be matched.
 */
[__AttributeUsage(_AttributeTargets.Var)]
public struct holoscan_alloc_size_ofAttribute
{
    string name;
};

/**
 * Specifies the grid dimensions for a compute kernel.
 * Defines how many threads will be launched in each dimension.
 *
 * @param x Number of threads in the X dimension
 * @param y Number of threads in the Y dimension
 * @param z Number of threads in the Z dimension
 */
[__AttributeUsage(_AttributeTargets.Function)]
public struct holoscan_grid_sizeAttribute
{
    int x;
    int y;
    int z;
};

/**
 * Specifies the grid dimensions for a compute kernel based on a resource's size.
 * The grid size is automatically calculated based on the dimensions of the specified resource.
 *
 * @param name The name of the resource whose dimensions should be used to calculate grid size.
 */
[__AttributeUsage(_AttributeTargets.Function)]
public struct holoscan_grid_size_ofAttribute
{
    string name;
};
