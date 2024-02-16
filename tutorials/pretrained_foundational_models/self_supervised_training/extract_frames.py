"""
SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import cv2
import numpy as np


def main(args):
    ROOT_DIR = args.data_path
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "./videos/"))
    VIDEO_NAMES = [x for x in VIDEO_NAMES if "mp4" in x]
    TRAIN_NUMBERS = np.arange(1, 41).tolist()
    VAL_NUMBERS = np.arange(41, 49).tolist()
    TEST_NUMBERS = np.arange(49, 81).tolist()

    for video_name in VIDEO_NAMES:
        print(video_name)
        vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "./videos/" + video_name))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("fps", fps)
        if fps != 25:
            print(video_name, "not at 25fps", fps)
        success = True
        count = 0
        vid_id = int(video_name.replace(".mp4", "").replace("video", ""))
        if vid_id in TRAIN_NUMBERS:
            save_dir = "./frames/train/" + video_name.strip(".mp4") + "/"
        elif vid_id in VAL_NUMBERS:
            save_dir = "./frames/val/" + video_name.strip(".mp4") + "/"
        elif vid_id in TEST_NUMBERS:
            save_dir = "./frames/test/" + video_name.strip(".mp4") + "/"
        save_dir = os.path.join(ROOT_DIR, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            while success is True:
                success, image = vidcap.read()
                if success:
                    cv2.imwrite(save_dir + str(count) + ".jpg", image)
                count += 1
        vidcap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # A path to save the result.
    parser.add_argument(
        "--datadir",
        type=str,
        default=r"/workspace/data",
        help="The path to dataset location.",
    )
    args = parser.parse_args()
    main(args)
