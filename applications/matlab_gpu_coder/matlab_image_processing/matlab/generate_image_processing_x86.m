% SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
% SPDX-License-Identifier: Apache-2.0
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
% http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

%
% Generates CUDA code and shared libraries on x86 devices
%

% Create configuration object of class 'coder.EmbeddedCodeConfig'.
cfg = coder.gpuConfig('dll','ecoder',true);

% Create a configuration object of class 'coder.CuDNNConfig'.
cfg.GenerateReport = true;
cfg.ReportPotentialDifferences = false;

% Define argument types for entry-point
in1 = coder.typeof(uint8(0),[480 854 3],'Gpu',true);
in2 = coder.typeof(single(0));

cfg.Hardware = coder.hardware('MATLAB Host Computer');

% Invoke MATLAB Coder
codegen -config cfg matlab_image_processing -args {in1, in2} -d ../codegen/dll/matlab_image_processing