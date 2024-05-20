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
% Generates CUDA code and shared libraries on Jetson devices
%

% Specify IP, username and p/w of device
clear hwobj
hwobj = jetson('ip','username','password');

% Create configuration object of class 'coder.EmbeddedCodeConfig'.
cfg = coder.gpuConfig('dll','ecoder',true);

% Create a configuration object of class 'coder.CuDNNConfig'.
cfg.GenerateReport = true;
cfg.ReportPotentialDifferences = false;
cfg.GpuConfig.ComputeCapability = '7.2';

% Define argument types for entry-point
in1 = coder.typeof(uint8(0),[480 854 3],'Gpu',true);
in2 = coder.typeof(single(0));

cfg.Hardware = coder.hardware('NVIDIA Jetson');
% Specify build directory on device
% The dll folder of this build directory is what should be pointed to in the CMakeLists.txt
cfg.Hardware.BuildDir = '<ABSOLUTE_PATH>/holohub/applications/matlab_image_processing';

% Invoke MATLAB Coder
codegen -config cfg matlab_image_processing -args {in1, in2}
