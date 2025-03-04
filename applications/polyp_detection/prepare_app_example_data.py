# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse
import os
import tarfile
import time

import cv2
import requests
from tqdm import tqdm


def download_file(url, local_filename, max_attempts=3, retry_delay=180):
    attempt = 0
    while attempt < max_attempts:
        try:
            with requests.get(url, stream=True, verify=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                block_size = 8192  # 8 KiB

                with open(local_filename, "wb") as f, tqdm(
                    total=total_size, unit="iB", unit_scale=True, desc=local_filename
                ) as bar:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return  # Exit the function if download is successful

        except Exception as e:
            print(f"Error occurred: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            attempt += 1


def extract_file(file_name, download_dir, extracted_folder_path):
    if not os.path.exists(extracted_folder_path):
        with tarfile.open(os.path.join(download_dir, file_name), "r") as tar_ref:
            tar_ref.extractall(download_dir)


def create_video_from_frames(num_frames, fps, output_name, frame_path):
    # Get a sorted list of frame filenames
    frame_files = sorted(
        [f for f in os.listdir(frame_path) if f.endswith(".jpg")],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    # Ensure num_frames does not exceed the available frames
    num_frames = min(num_frames, len(frame_files))

    # Read the first frame to get the dimensions
    first_frame = cv2.imread(os.path.join(frame_path, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    # Write the frames to the video
    # In 001-014_frames, frames from 21219 has lesions.
    # Therefore, we prepared a sample video from 21200 to 21200 + num_frames (defaults to 900)
    # This is to ensure that the sample video has lesions.
    start_frame = 21200
    for i in range(start_frame, start_frame + num_frames):
        frame = cv2.imread(os.path.join(frame_path, frame_files[i]))
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video {output_name} created successfully with {num_frames} frames at {fps} fps.")


def main():
    """
    This script downloads a sample of the REAL-Colon dataset from figshare and extracts it.
    The entire code is based on the implementation found at the following reference:
    https://github.com/cosmoimd/real-colon-dataset/blob/main/figshare_dataset.py

    The dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
    For more details, visit: https://creativecommons.org/licenses/by/4.0/
    """

    parser = argparse.ArgumentParser(description="Download and extract REAL-Colon dataset.")
    parser.add_argument(
        "--download_dir", type=str, default="real_colon", help="Directory to download files to."
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="001-014_frames.tar.gz",
        help="Name of the file to download.",
    )
    parser.add_argument(
        "--video_name", type=str, default="001-014_sample.mp4", help="Name of the video to save."
    )
    parser.add_argument(
        "--video_frames", type=int, default=900, help="Number of frames to sample from the video."
    )
    args = parser.parse_args()

    article_url = "https://api.figshare.com/v2/articles/22202866"
    response = requests.get(article_url, verify=True)
    response.raise_for_status()
    article_data = response.json()

    os.makedirs(args.download_dir, exist_ok=True)

    file_info = next(
        (item for item in article_data["files"] if item["name"] == args.file_name), None
    )
    if file_info:
        file_url = file_info["download_url"]
        download_path = os.path.join(args.download_dir, args.file_name)
        extracted_folder_name = os.path.splitext(
            os.path.basename(args.file_name.rstrip(".tar.gz"))
        )[0]
        extracted_folder_path = os.path.join(args.download_dir, extracted_folder_name)
        if not os.path.exists(extracted_folder_path):
            download_file(file_url, download_path)
            extract_file(args.file_name, args.download_dir, extracted_folder_path)
        else:
            print(f"File {args.file_name} already exists in the download directory.")

        if not os.path.exists(os.path.join(args.download_dir, args.video_name)):
            create_video_from_frames(
                args.video_frames,
                30,
                os.path.join(args.download_dir, args.video_name),
                extracted_folder_path,
            )
        else:
            print(f"Video {args.video_name} already exists in the download directory.")
    else:
        print(f"File {args.file_name} not found in the article data.")


if __name__ == "__main__":
    main()
