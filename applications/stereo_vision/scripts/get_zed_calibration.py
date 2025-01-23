"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import configparser
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

rez = "FHD"
calibration_file = "calibration.yaml"

parser = argparse.ArgumentParser(description="Get factory ZED calibration")
required = parser.add_argument_group("required arguments")
required.add_argument(
    "-s", "--serial_number", help="serial number of the ZED camera", required=True
)
required.add_argument(
    "-o",
    "--output",
    help="output calibration file (default: stereo_calibration.yaml)",
    default="stereo_calibration.yaml",
)
required.add_argument("-r", "--resolution", help="resolution setting (default: FHD)", default="FHD")
args = parser.parse_args()

serial_number = args.serial_number
rez = args.resolution
calibration_file = args.output
if rez == "2K":
    width = 2208
    height = 1242
elif rez == "FHD":
    width = 1920
    height = 1080
elif rez == "HD":
    width = 1280
    height = 720
elif rez == "VGA":
    width = 672
    height = 376
else:
    raise NameError("resolution not supported")


tmp_file = str(serial_number) + ".tmp"
url = "http://calib.stereolabs.com/?SN=" + str(serial_number)
os.system("wget -O" + tmp_file + " " + url)

calib = configparser.ConfigParser()
calib.read(tmp_file)
M1 = np.zeros([3, 3])
M1[0, 0] = calib["LEFT_CAM_" + rez]["fx"]
M1[0, 2] = calib["LEFT_CAM_" + rez]["cx"]
M1[1, 1] = calib["LEFT_CAM_" + rez]["fy"]
M1[1, 2] = calib["LEFT_CAM_" + rez]["cy"]
M1[2, 2] = 1.0

d1 = np.zeros(5)
d1[0] = calib["LEFT_CAM_" + rez]["k1"]
d1[1] = calib["LEFT_CAM_" + rez]["k2"]
d1[2] = calib["LEFT_CAM_" + rez]["p1"]
d1[3] = calib["LEFT_CAM_" + rez]["p2"]
d1[4] = calib["LEFT_CAM_" + rez]["k3"]

M2 = np.zeros([3, 3])
M2[0, 0] = calib["RIGHT_CAM_" + rez]["fx"]
M2[0, 2] = calib["RIGHT_CAM_" + rez]["cx"]
M2[1, 1] = calib["RIGHT_CAM_" + rez]["fy"]
M2[1, 2] = calib["RIGHT_CAM_" + rez]["cy"]
M2[2, 2] = 1.0

d2 = np.zeros(5)
d2[0] = calib["RIGHT_CAM_" + rez]["k1"]
d2[1] = calib["RIGHT_CAM_" + rez]["k2"]
d2[2] = calib["RIGHT_CAM_" + rez]["p1"]
d2[3] = calib["RIGHT_CAM_" + rez]["p2"]
d2[4] = calib["RIGHT_CAM_" + rez]["k3"]

t = np.zeros(3)
t[0] = -float(calib["STEREO"]["Baseline"])
t[1] = calib["STEREO"]["TY"]
t[2] = calib["STEREO"]["TZ"]

r = np.zeros(3)
r[0] = calib["STEREO"]["RX_" + rez]
r[1] = calib["STEREO"]["CV_" + rez]
r[2] = calib["STEREO"]["RZ_" + rez]
R = Rotation.from_rotvec(r).as_matrix()

calibration = dict(
    SN=serial_number,
    M1=str(M1.flatten().tolist()),
    d1=str(d1.tolist()),
    M2=str(M2.flatten().tolist()),
    d2=str(d2.tolist()),
    t=str(t.tolist()),
    R=str(R.flatten().tolist()),
    width=width,
    height=height,
)

output_yaml = yaml.dump(
    calibration, default_style=None, default_flow_style=False, sort_keys=False
).replace("'", "")
print("Writing calibration to: " + calibration_file)
with open(calibration_file, "w") as f:
    f.write(output_yaml)

os.system("rm " + tmp_file)
