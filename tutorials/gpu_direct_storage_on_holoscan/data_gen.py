"""
 SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse

import cupy as cp
import kvikio
from scipy.stats import poisson

parser = argparse.ArgumentParser(
    prog="holoscan_gds.py",
    description="Example of GPU Direct Storage on IGX",
    epilog="See <Example on github> for more information",
)

parser.add_argument("--out_path", help="Path for file output", default="/mnt/nvme/data/test-file")
parser.add_argument("--shape1", help="Dimension 1 for data", default=256)
parser.add_argument("--shape2", help="Dimension 2 for data", default=256)
parser.add_argument("--shape3", help="Dimension 3 for data", default=20000)
args = parser.parse_args()


def main():
    shape = (int(args.shape1), int(args.shape2), int(args.shape3))

    # credit to emdfile: https://github.com/py4dstem/emdfile/blob/main/tutorials/emdfile_intro_example.ipynb
    class DataGenerator:
        """This class generates data by
        (1) initializing some starting position `xy0` and velocity `v0`
        (2) finding the position at each of 8 time points
        (3) at each of these 8 time points, place a 2D Gaussian into a 256x256 grid
        (4) generate a 256x256 image at each time point by drawing from a Poisson
            distribution, using an expected value given at each pixel by the Gaussian
        """

        params = {"A": 5, "sigx": 12, "sigy": 9}  # 2D gaussian parameters
        mu_scale = 1  # scaling for Poisson draws

        def __init__(self, xy0, v0):
            """2-tuples xy0 and v0"""

            # set the initial position and velocity
            self.xy0 = cp.array(xy0)
            self.v0 = cp.array(v0)

            # find the center at each time point
            self.xy = (
                self.xy0[:, cp.newaxis]
                + cp.tile(cp.arange(shape[-1]), (2, 1)) * self.v0[:, cp.newaxis]
            )

            # make the data
            self.generate_data()

        def generate_data(self):
            """make a 2D gaussian on a grid, then draw from a Poisson distribution using
            an expected value given at each pixel by the Gaussian
            """

            # make a meshgrid
            yy, xx = cp.meshgrid(cp.arange(shape[1]), cp.arange(shape[0]))
            self.xx, self.yy = xx - shape[0] / 2.0, yy - shape[1] / 2.0

            # extend in the third dimension and center
            self.xx = cp.dstack([self.xx[:, :, cp.newaxis] - x for x in self.xy[0, :]])
            self.yy = cp.dstack([self.yy[:, :, cp.newaxis] - y for y in self.xy[1, :]])

            # get the data
            self.data = poisson.rvs(
                self.gaussian((self.xx, self.yy), **(DataGenerator.params)) * DataGenerator.mu_scale
            )

        @staticmethod
        def gaussian(p, A, sigx, sigy):
            return A * cp.exp(-(0.5 * (p[0] / sigx) ** 2)) * cp.exp(-(0.5 * (p[1] / sigy) ** 2))

    # generate some data
    xy0 = (-17, -9)
    v0 = (6, 8)
    data_generator = DataGenerator(xy0, v0)

    a = cp.array(data_generator.data)
    f = kvikio.CuFile(args.out_path, "w")
    # Write whole array to file
    f.write(a)
    f.close()
