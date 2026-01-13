# MIT License
#
# Copyright (c) 2025-2026, EndoGaussian Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Scene rendering and visualization utilities for training.
MIT-licensed implementation derived from EndoGaussian project.
"""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Configure matplotlib fonts
plt.rcParams["font.sans-serif"] = ["Times New Roman"]


@torch.no_grad()
def render_training_image(
    scene, gaussians, viewpoints, render_func, pipe, background, stage, iteration, time_now
):
    def render(gaussians, viewpoint, path, scaling):
        render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage)
        label1 = f"stage:{stage},iter:{iteration}"
        times = time_now / 60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert channel order to (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()

        # Normalize depth (prevent division by zero)
        depth_max = depth_np.max()
        if depth_max > 0:
            depth_np /= depth_max
        # If depth_max == 0, depth_np remains all zeros (valid visualization)

        depth_np = np.repeat(depth_np, 3, axis=2)
        image_np = np.concatenate((image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray(
            (np.clip(image_np, 0, 1) * 255).astype("uint8")
        )  # Convert to 8-bit image
        # Create a copy of PIL image object to draw labels
        draw1 = ImageDraw.Draw(image_with_labels)

        # Select font and font size
        font = ImageFont.load_default()  # Use default font

        # Select text color
        text_color = (255, 0, 0)  # Red

        # Select label positions (top-left and top-right coordinates)
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # Top-right corner

        # Add labels to the image
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)

        image_with_labels.save(path)

    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path, "pointclouds")
    image_path = os.path.join(render_base_path, "images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Render and save images for all viewpoints
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path, f"{iteration}_{idx}.jpg")
        render(gaussians, viewpoints[idx], image_save_path, scaling=1)

    # Extract point cloud for debugging (unused currently)
    # pc_mask = gaussians.get_opacity
    # pc_mask = pc_mask > 0.1
    # xyz = gaussians.get_xyz.detach()[pc_mask.squeeze()].cpu().permute(1,0).numpy()


def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    R = R.T
    # Apply rotation and translation transform
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)

    # Visualize point cloud
    ax.scatter(
        transformed_point_cloud[0],
        transformed_point_cloud[1],
        transformed_point_cloud[2],
        c="g",
        marker="o",
    )
    ax.axis("off")

    # Save rendered result as image
    plt.savefig(filename)
    plt.close(fig)  # Close figure to prevent memory leak
