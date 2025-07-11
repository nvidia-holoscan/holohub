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

import numpy as np
from osgeo import gdal
from shapely.geometry import MultiPoint


def ray_plane_intersections_cpu(planeNormal, planePoint, rayDirections, rayPoint, epsilon=1e-6):
    # ndotu = planeNormal.dot(rayDirections[0,:])
    npoints = rayDirections.shape[0]
    planeNormals = np.repeat(planeNormal[np.newaxis, :], npoints, axis=0)
    ndotus = (planeNormals * rayDirections).sum(1)
    if (np.abs(ndotus) < epsilon).any():
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    # si = -planeNormal.dot(w) / ndotu

    ws = np.repeat(w[np.newaxis, :], npoints, axis=0)
    sis = -(planeNormals * ws).sum(1) / ndotus

    # Psi = w + si * rayDirections[0,:] + planePoint
    sis = np.repeat(sis[:, np.newaxis], 3, axis=1)
    inters = ws + sis * rayDirections + planePoint

    return inters


def extract_extent_nativeCRS(image_fname):
    info = gdal.Info(image_fname, options=gdal.InfoOptions(format="json", stats=True))
    top_left = [float(item) for item in info["cornerCoordinates"]["upperLeft"]]
    bot_left = [float(item) for item in info["cornerCoordinates"]["lowerLeft"]]
    top_right = [float(item) for item in info["cornerCoordinates"]["upperRight"]]
    bot_right = [float(item) for item in info["cornerCoordinates"]["lowerRight"]]
    corners = np.vstack([bot_right, bot_left, top_left, top_right, bot_right])
    bpts = MultiPoint(list(corners))
    return bpts.convex_hull.bounds


def load_basemap(basemap_fname, nx, ny, bands=[1, 2, 3]):
    warp_ops = gdal.WarpOptions(
        width=nx, height=ny, resampleAlg="cubic", creationOptions=["COMPRESS=LZW"]
    )
    warped_fname = os.path.splitext(basemap_fname)[0] + "_resized.tif"
    dset = gdal.Warp(warped_fname, basemap_fname, options=warp_ops)
    img = []
    for bn in bands:
        band = dset.GetRasterBand(bn)
        frame = band.ReadAsArray()
        img.append(np.flipud(frame))
    dset = None

    return np.dstack(img)


def create_blank_basemap(nx, ny, bands=3):
    return np.zeros((ny, nx, bands), dtype=np.uint8)
