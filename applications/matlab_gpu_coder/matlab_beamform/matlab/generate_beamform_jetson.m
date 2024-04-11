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
ARGS = cell(1,1);
ARGS{1} = cell(4,1);
ARGS{1}{1} = coder.typeof(single(1i),[6494 4000],'Gpu',true);
ARGS_1_2 = struct;
ARGS_1_2.c = coder.typeof(0);
ARGS_1_2.fc = coder.typeof(0);
ARGS_1_2.rangeRes = coder.typeof(0);
ARGS_1_2.alongTrackRes = coder.typeof(0);
ARGS_1_2.Bw = coder.typeof(0);
ARGS_1_2.prf = coder.typeof(0);
ARGS_1_2.speed = coder.typeof(0);
ARGS_1_2.aperture = coder.typeof(0);
ARGS_1_2.Tpd = coder.typeof(0);
ARGS_1_2.fs = coder.typeof(0);
ARGS{1}{2} = coder.typeof(ARGS_1_2);
ARGS{1}{3} = coder.typeof(single(0),[1 4000],'Gpu',true);
ARGS{1}{4} = coder.typeof(single(0),[1 6494],'Gpu',true);

cfg.Hardware = coder.hardware('NVIDIA Jetson');
% Specify build directory on device
% The dll folder of this build directory is what should be pointed to in the CMakeLists.txt
cfg.Hardware.BuildDir = '<ABSOLUTE_PATH>/holohub/applications/matlab_gpu_coder';

% Invoke MATLAB Coder
codegen -config cfg matlab_beamform -args ARGS{1}