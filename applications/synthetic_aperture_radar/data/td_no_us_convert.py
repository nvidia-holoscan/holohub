# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#import cupy as cp
import pickle

import numpy as np
import scipy.io as sio


def npot(x):
    return 1<<(x-1).bit_length()

files=[ \
"./Gotcha-Large-Scene-Data-Disk-1/subData/subData01.mat", \
"./Gotcha-Large-Scene-Data-Disk-1/subData/subData02.mat", \
"./Gotcha-Large-Scene-Data-Disk-1/subData/subData03.mat", \
"./Gotcha-Large-Scene-Data-Disk-1/subData/subData04.mat", \
"./Gotcha-Large-Scene-Data-Disk-1/subData/subData05.mat", \
"./Gotcha-Large-Scene-Data-Disk-2/subData/subData06.mat", \
"./Gotcha-Large-Scene-Data-Disk-2/subData/subData07.mat", \
"./Gotcha-Large-Scene-Data-Disk-2/subData/subData08.mat", \
"./Gotcha-Large-Scene-Data-Disk-2/subData/subData09.mat", \
"./Gotcha-Large-Scene-Data-Disk-2/subData/subData10.mat", \
]

Ant = None
Np = None
phdata = None
R0 = None


for f in files:
  print (f)
  data = sio.loadmat(f)
  #print (data['subData'])
  X=data['subData']
  this_Np     = X['Np'][0][0].flatten()
  this_K      = X['K'][0][0][0][0]
  this_df     = X['deltaF'][0][0][0][0]
  this_minf   = X['minF'][0][0][0][0]
  this_AntX   = X['AntX'][0][0].flatten()
  this_AntY   = X['AntY'][0][0].flatten()
  this_AntZ   = X['AntZ'][0][0].flatten()
  this_R0     = X['R0'][0][0].flatten()
  this_phdata = X['phdata'][0][0].transpose()
  this_Ant = np.stack([this_AntX, this_AntY, this_AntZ], axis=1)
  print ("this_Np.shape=", this_Np.shape)
  print ("this_AntX.shape=", this_AntX.shape)
  print ("this_AntY.shape=", this_AntY.shape)
  print ("this_AntZ.shape=", this_AntZ.shape)
  print ("this_Ant.shape=", this_Ant.shape)
  print ("this_R0.shape=", this_R0.shape)
  print ("this_phdata.shape=", this_phdata.shape)
  print ("this_phdata.dtype=", this_phdata.dtype)


  sample_count = this_phdata.shape[1]
  print ("sample count=", sample_count)

  #padded_sample_count = npot(sample_count)
  padded_sample_count = sample_count
  print ("padded count=", padded_sample_count)

  pad_count = padded_sample_count - sample_count
  print ("pad count=", pad_count)

  this_phdata = np.pad (this_phdata, ((0,0),(0,pad_count)), 'constant', constant_values = ((0,0),(0,0)))
  print ("this_phdata.shape=", this_phdata.shape)

  this_phdata_a = np.fft.ifft (this_phdata, norm="forward")
  this_phdata = np.fft.fftshift (this_phdata_a, axes=(1,))

  C = np.float64(299792458.0)
  df = this_df
  print ("df=", df)

  K = this_K
  df = this_df
  minf = this_minf

  calc_r0 = np.sqrt (this_AntX * this_AntX + this_AntY * this_AntY + this_AntZ * this_AntZ)
  r0_diff = this_R0 - calc_r0
  print ("R0 error norm is ", np.linalg.norm(r0_diff))
  print ("K = ", K, " df = ", df, " minf = ", minf, " R0 = ", R0)
  if Ant is None:
      Ant = this_Ant
  else:
      Ant = np.concatenate ((Ant, this_Ant), axis=0)
  print ("Ant.shape=", Ant.shape)

  if R0 is None:
      R0 = this_R0
  else:
      R0 = np.concatenate ((R0, this_R0), axis=0)
  print ("R0.shape=", R0.shape)

  if Np is None:
      Np = this_Np
  else:
      Np = np.concatenate((Np, this_Np), axis=0)
  print ("Np.shape=", Np.shape)

  if phdata is None:
      phdata = this_phdata
  else:
      phdata = np.concatenate((phdata, this_phdata), axis=0)
  print ("phdata.shape=", phdata.shape)

print (Np)

mdic = {"Np" : Np, "Antenna": Ant, "phase_data": phdata}

#with open("gotcha_large_scene.dat", 'wb') as pfile:
#    #pickle.dump(mdic, pfile, protocol=pickle.HIGHEST_PROTOCOL)

with open ("gotcha_binary-td.dat", "wb") as bfile:
    num_pulses = np.array(phdata.shape[0]).astype(np.uint32)
    bfile.write(num_pulses.astype(np.int32))

    num_samples = np.array(phdata.shape[1]).astype(np.uint32)
    bfile.write(num_samples.astype(np.int32))

    minf =np.array (minf)
    bfile.write (minf)

    #df =np.array (df)
    #bfile.write (df)

    bfile.write (df) 

    for p in range(phdata.shape[0]):
        bfile.write (Ant[p,:].astype(np.float32))
        bfile.write (R0[p].astype (np.float64))
        bfile.write (phdata[p,:].astype(np.complex64))
    bfile.close()
