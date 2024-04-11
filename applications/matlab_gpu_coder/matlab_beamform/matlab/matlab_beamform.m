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

function sdata = matlab_beamform(cdata, procParam, xAxis, fastTime) %#codegen

sdata = cdata;
tiny = 1e-4;

coder.gpu.kernelfun();

for jj = 1: numel(xAxis)
    for ii = 1: numel(fastTime)

        % Using desired along track resolution, compute the synthetic aperture length (no check implemented to prevent Lsar for exceeding pixel index range)
        Lsynth= (procParam.c/procParam.fc)* (procParam.c*fastTime(ii)/2)/(2*procParam.alongTrackRes);
        Lsar = round(Lsynth*length(xAxis)/(xAxis(end)-xAxis(1))) ;
        Lsar = Lsar + mod(Lsar,2);     % Ensuring Lsar is an even number
        signal = complex(single(0),single(0));

        %Initialize window length of aperture
        % hn = 0.5*(1-cos(2*pi*(1:Lsar)/Lsar));
        pos_x= xAxis(jj);
        pos_y= procParam.c*fastTime(ii)/2;
        count= complex(single(0),single(0));

        if ((jj-Lsar/2+1) > 0 && (jj+Lsar/2) < length(xAxis))
            for k= jj-Lsar/2 +1 :jj+ Lsar/2 % Iterate over the synthetic aperture
                td= sqrt((xAxis(k)- pos_x)^2 + pos_y^2)*2/procParam.c;
                cell= round(real(td*procParam.fs)) +1 ;

                if cell<length(fastTime)
                    signal = cdata(cell, k);
                end

                count = count + signal*exp(1j*2*pi*procParam.fc*(td));
            end
        end

        sdata(ii,jj)= count;
    end
end

maxVal = gpucoder.reduce(abs(sdata), @myMax) + tiny;
sdata = 20*log10(abs(sdata./maxVal) + tiny);

sdata = uint8(normalize(sdata, 'range', [0 255]));

% Until Holoviz supports single-channel uint8 data we have to output
% an RGB image
sdata = cat(3, sdata, sdata, sdata);

end

function out = myMax(in1, in2)
    out = max(in1, in2);
end