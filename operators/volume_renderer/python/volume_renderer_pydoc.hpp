/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_VOLUME_RENDERER_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_VOLUME_RENDERER_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace VolumeRendererOp {

PYDOC(VolumeRendererOp, R"doc(
The `volume_renderer` operator reads 3D volumes from the specified input file.

The operator supports these file formats:
* MHD https://itk.org/Wiki/ITK/MetaIO/Documentation
* NIFTI https://nifti.nimh.nih.gov/
* NRRD https://teem.sourceforge.net/nrrd/format.html
)doc")

// PyVolumeRendererOp Constructor
PYDOC(VolumeRendererOp_python, R"doc(
Operator class to render a volume.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
config_file : str, optional
    Config file path. The content of the file is passed to `clara::viz::JsonInterface::SetSettings()` at initialization time.
write_config_file : str, optional
    Deduce config settings from volume data and write to file. Sets a light in correct distance.
    Sets a transfer function using the histogram of the data. Writes the JSON configuration to the
    file with the given name.
allocator : ``holoscan.resources.Allocator``, optional
    Allocator used to allocate render buffer outputs when no pre-allocated color or depth buffer is passed to `color_buffer_in` or `depth_buffer_in`. Allocator needs to be capable to allocate device memory.
alloc_width : int, optional
    Width of the render buffer to allocate when no pre-allocated buffers are provided.
alloc_height : int, optional
    Height of the render buffer to allocate when no pre-allocated buffers are provided.
density_min : float, optional
    Minimum density volume element value. If not set this is calculated from the volume data. In
    practice CT volumes have a minimum value of -1024 which corresponds to the lower value of the
    Hounsfield scale range usually used.
density_max : float, optional
    Maximum density volume element value. If not set this is calculated from the volume data. In
    practice CT volumes have a minimum value of -1024 which corresponds to the lower value of the
    Hounsfield scale range usually used.
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

}  // namespace VolumeRendererOp


}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VOLUME_RENDERER_PYDOC_HPP
