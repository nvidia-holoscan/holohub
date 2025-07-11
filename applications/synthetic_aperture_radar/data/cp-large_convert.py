# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

#import cupy as cp
import pickle

import numpy as np
import scipy.io as sio

prefix="./GOTCHA-CP_Disc1/DATA/pass1/HH"
output_filename="gotcha-cp-td-os.dat"

def npot(x):
    return 1<<(x-1).bit_length()


files = os.listdir (prefix)
p2 = prefix + '/{0}'
files = [p2.format(i) for i in files]
files.sort()
print ("files=", files)

Ant = None
Np = None
fp = None
R0 = None


for f in files:
  print (f)
  data = sio.loadmat(f)
  print ("keys=", data.keys())
  #print (data['subData'])
  X=data['data']
  #this_Np     = X['Np'][0][0].flatten()
  #this_K      = X['K'][0][0][0][0]
  #this_df     = X['deltaF'][0][0][0][0]
  #this_minf   = X['minF'][0][0][0][0]
  this_AntX   = X['x'][0][0].flatten()
  this_AntY   = X['y'][0][0].flatten()
  this_AntZ   = X['z'][0][0].flatten()
  this_freq   = X['freq'][0][0].flatten()
  this_R0     = X['r0'][0][0].flatten()
  this_fp     = X['fp'][0][0].transpose()
  this_Ant = np.stack([this_AntX, this_AntY, this_AntZ], axis=1)
  print ("this_AntX.shape=", this_AntX.shape)
  print ("this_AntY.shape=", this_AntY.shape)
  print ("this_AntZ.shape=", this_AntZ.shape)
  print ("this_Ant.shape=", this_Ant.shape)
  #print ("this_R0.shape=", this_R0.shape)
  print ("this_fp.shape=", this_fp.shape)
  print ("this_fp.dtype=", this_fp.dtype)


  sample_count = this_fp.shape[1]
  pulses_count = this_fp.shape[0]
  print ("sample count=", sample_count)
  print ("pulses count=", pulses_count)


  padded_sample_count = npot(sample_count)*16
  #padded_sample_count = sample_count
  print ("padded count=", padded_sample_count)

  pad_count = padded_sample_count - sample_count
  print ("pad count=", pad_count)

  this_phdata = np.pad (this_fp,     ((0,0),(0,pad_count)), 'constant', constant_values = ((0,0),(0,0)))
  print ("this_phdata.shape=", this_phdata.shape)

  this_phdata = np.fft.ifft (this_phdata, norm="forward")
  this_phdata = np.fft.fftshift (this_phdata, axes=(1,))
  this_fp = this_phdata

  
  C = np.float64(299792458.0)

  df = this_freq[1] - this_freq[0]
  print ("df=", df)
  minf = this_freq[0]

  calc_r0 = np.sqrt (this_AntX * this_AntX + this_AntY * this_AntY + this_AntZ * this_AntZ)
  r0_diff = this_R0 - calc_r0
  print ("R0 error norm is ", np.linalg.norm(r0_diff))
  print (" df = ", df, " minf = ", minf, " R0 = ", this_R0)
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

  if fp is None:
      fp = this_fp
  else:
      fp = np.concatenate((fp, this_fp), axis=0)
  print ("fp.shape=", fp.shape)

with open (output_filename, "wb") as bfile:
    num_pulses = np.array(fp.shape[0]).astype(np.uint32)
    bfile.write(num_pulses.astype(np.int32))

    num_samples = np.array(fp.shape[1]).astype(np.uint32)
    bfile.write(num_samples.astype(np.int32))

    minf =np.array (minf)
    print ("minf=", minf)
    bfile.write (minf.astype(np.float64))

    df = np.array(df)
    print ("df=", df)
    bfile.write (df.astype(np.float64)) 

    #df =np.array (df)
    #bfile.write (df)


    for p in range(fp.shape[0]):
        bfile.write (Ant[p,:].astype(np.float32))
        bfile.write (R0[p].astype (np.float64))
        bfile.write (fp[p,:].astype(np.complex64))
    bfile.close()
