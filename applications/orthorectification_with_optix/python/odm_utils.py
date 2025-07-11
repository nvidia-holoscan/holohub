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
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
import tqdm
from numpy.linalg import norm
from odm_report_shot_coverage.models.reconstruction import parse_reconstruction
from ortho_utils import extract_extent_nativeCRS, ray_plane_intersections_cpu
from osgeo import gdal
from shapely.geometry import MultiPoint, Point, box


def norm2uv_odm(norm_coords, shot):
    x_n, y_n = norm_coords
    r_2 = x_n * x_n + y_n * y_n
    d = 1 + r_2 * shot.camera.k1 + r_2 * r_2 * shot.camera.k2
    return shot.camera.focal * d * x_n, shot.camera.focal * d * y_n


def load_frame_odm(
    reconstruction,
    offsets,
    mosaic_bbox,
    use_mosaic_bbox,
    data_dir,
    sensor_resize,
    sensor_dims,
    min_el,
    fnum,
):
    shot = reconstruction.shots[fnum]
    utm_sensor_pos = [
        shot.translation[0] + offsets[0],
        shot.translation[1] + offsets[1],
        shot.translation[2],
    ]

    utm_sensor_pos_str = ",".join([str(item) for item in utm_sensor_pos])

    uvx = norm2uv_odm((0.5, 0), shot)[0]
    scale_ratio = shot.camera.height / shot.camera.width
    uvy = norm2uv_odm((0, scale_ratio * 0.5), shot)[1]

    hfov = 2 * np.rad2deg(np.arctan((uvx / shot.camera.focal) / shot.camera.focal))
    vfov = 2 * np.rad2deg(np.arctan((uvy / shot.camera.focal) / shot.camera.focal))

    # to get direction vector - start with camera coords at x,y,z = (0,0,1), then rotate
    rc = [0, 0, 1]
    sensor_dir = shot._transfo_rotation.apply(rc, inverse=True)

    # to get sensor up vector - start with camera coords at x,y,z = (0, -1, 0), then rotate back
    rc = [0, -1, 0]
    sensor_up = shot._transfo_rotation.apply(rc, inverse=True)

    sensor_dir_str = ",".join([str(item) for item in sensor_dir])
    sensor_up_str = ",".join([str(item) for item in sensor_up])

    # use camera model to get bounding box estimates in world coordinates assuming
    # flat plane elevation with min elevation
    if not use_mosaic_bbox:
        # vector in direction of corners
        x_edge = shot.camera.focal * np.tan(np.deg2rad(hfov / 2.0))
        y_edge = shot.camera.focal * np.tan(np.deg2rad(vfov / 2.0))

        bot_right = np.array(
            shot._transfo_rotation.apply([x_edge, y_edge, shot.camera.focal], inverse=True)
        )
        bot_right = bot_right / norm(bot_right)

        bot_left = np.array(
            shot._transfo_rotation.apply([-x_edge, y_edge, shot.camera.focal], inverse=True)
        )
        bot_left = bot_left / norm(bot_left)

        top_left = np.array(
            shot._transfo_rotation.apply([-x_edge, -y_edge, shot.camera.focal], inverse=True)
        )
        top_left = top_left / norm(top_left)

        top_right = np.array(
            shot._transfo_rotation.apply([x_edge, -y_edge, shot.camera.focal], inverse=True)
        )
        top_right = top_right / norm(top_right)

        corners = np.vstack([bot_right, bot_left, top_left, top_right, bot_right])

        utm_corners = ray_plane_intersections_cpu(
            np.array([0, 0, 1]), np.array([0, 0, min_el]), corners, np.array(utm_sensor_pos)
        )

        bpts = MultiPoint(list(utm_corners))
        bbox = bpts.convex_hull
    else:
        bbox = mosaic_bbox

    camera_loc = Point(utm_sensor_pos)

    meta = {
        "wkt": bbox.wkt,
        "wkt_camera_loc": camera_loc.wkt,
        "image_name": shot.image_name,
        "sensor_dir": sensor_dir_str,
        "sensor_up": sensor_up_str,
        "utm_pos": utm_sensor_pos_str,
        "hfov": str(hfov),
        "vfov": str(vfov),
        "frame_number": fnum,
    }

    sensor_image_fname = os.path.join(data_dir, "images", meta["image_name"])
    sensor_pix = cv2.imread(sensor_image_fname)
    if sensor_resize is not None:
        sensor_pix = cv2.resize(sensor_pix, dsize=sensor_dims, interpolation=cv2.INTER_AREA)
    sensor_pix = cv2.cvtColor(sensor_pix, cv2.COLOR_BGR2RGB)

    return {"sensor_pix": sensor_pix, "meta": meta}


def load_all_frames_odm(
    use_mosaic_bbox,
    dem_image_fname,
    geo_path,
    data_dir,
    iterations,
    ncpu,
    sensor_resize,
    sensor_dims,
):
    dem_info = gdal.Info(dem_image_fname, options=gdal.InfoOptions(format="json", stats=True))
    min_el = float(dem_info["bands"][0]["metadata"][""]["STATISTICS_MINIMUM"])

    with open(geo_path) as fp:
        _ = fp.readline()
        offsets = [int(item) for item in fp.readline().strip().split(" ")]

    reconstruction = parse_reconstruction(data_dir)

    bbox = None
    if use_mosaic_bbox:
        bbox_bounds = extract_extent_nativeCRS(dem_image_fname)
        bbox = box(bbox_bounds[0], bbox_bounds[1], bbox_bounds[2], bbox_bounds[3])

    staged_dataset = {}
    frame_nums = list(range(iterations))
    func = partial(
        load_frame_odm,
        reconstruction,
        offsets,
        bbox,
        use_mosaic_bbox,
        data_dir,
        sensor_resize,
        sensor_dims,
        min_el,
    )

    pool = Pool(ncpu)
    for item in tqdm.tqdm(
        pool.imap(func, frame_nums), desc="simulating sensor feed", total=iterations
    ):
        staged_dataset[item["meta"]["frame_number"]] = item
    pool.close()
    pool.join()

    return staged_dataset
