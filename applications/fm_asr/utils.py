# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp
import yaml


def load_params(*args):
    # Load default parameters
    param_dir = os.path.join("/workspace/params")
    with open(os.path.join(param_dir, "default_params.yml")) as f:
        params = yaml.safe_load(f)

    # Load in over-write directories
    for param_file in args:
        try:
            param_file = os.path.join(param_dir, param_file)
            assert param_file[-3:] == "yml" or param_file[-4:] == "yaml"
            with open(param_file) as f:
                overwrite_params = yaml.safe_load(f)
            params = {**params, **overwrite_params}  # merge
        except TypeError:
            print(f"Not able to make a path out of {param_file}, ignoring")
        except FileNotFoundError:
            print(f"{param_file} is not an existing parameter file, ignoring")
        except AssertionError:
            print(f"{param_file} is not a YAML file, ignoring")

    return params


def float_to_pcm(f_data: cp.array, dtype=cp.int16):
    """
    Function made using the following sources:
        https://stackoverflow.com/questions/15087668/how-to-convert-pcm-samples-in-byte-array-as-floating-point-numbers-in-the-range
        http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
        https://stackoverflow.com/questions/51486093/convert-pcm-wave-data-to-numpy-arrays-and-vice-versa
    """
    dtype_max = cp.iinfo(dtype).max
    dtype_min = cp.iinfo(dtype).min
    abs_int_max = 2 ** (cp.iinfo(dtype).bits - 1)
    return cp.clip(f_data * abs_int_max, dtype_min, dtype_max).astype(cp.int16)


def run_time_to_iterations(run_time, src_fs, buffer_size):
    """
    Calculates the number of source operator iterations needed to achieve
        the requested runtime. Intended to be used to instantiate
        CountConditions.
    run_time: desired pipeline runtime (seconds)
    src_fs: sample rate of the datasource (samples/second)
    buffer_size: size of input buffer (number of entries, not bytes)
    """
    total_samples = run_time * src_fs
    total_iters = total_samples // buffer_size
    return total_iters
