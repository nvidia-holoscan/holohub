# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import filecmp
import unittest

import numpy as np
from PIL import Image
from sam2operator import ImagePredictorProcessor
from utils import show_masks


class TestSam2(unittest.TestCase):
    def test_sam2(self):
        sam2_checkpoint = "/workspace/segment-anything-2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        image_path = "/workspace/segment-anything-2/notebooks/images/cars.jpg"
        save_dir = "/workspace/holohub/applications/sam2/tests"

        # Load image
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        # Set input point and label
        input_point = np.array([[500, 375]])
        input_label = np.array([1])

        # Initialize ImagePredictorProcessor
        processor = ImagePredictorProcessor(sam2_checkpoint, model_cfg)

        # Compute masks, scores, and logits
        masks, scores, logits = processor.compute(image, input_point, input_label)
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # Show masks and save the result
        show_masks(
            image,
            masks,
            scores,
            point_coords=input_point,
            input_labels=input_label,
            borders=True,
            save_dir=save_dir,
        )

        # compare the saved image with the expected image
        expected_image_path = "/workspace/holohub/applications/sam2/tests/expected/mask_0.png"
        self.assertTrue(filecmp.cmp(expected_image_path, f"{save_dir}/mask_0.png"))


class TestImagePredictorProcessor(unittest.TestCase):
    def setUp(self):
        sam2_checkpoint = "/workspace/segment-anything-2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        image_path = "/workspace/segment-anything-2/notebooks/images/cars.jpg"

        # Load image
        image = Image.open(image_path)
        self.image = np.array(image.convert("RGB"))

        # Initialize ImagePredictorProcessor
        self.processor = ImagePredictorProcessor(sam2_checkpoint, model_cfg)

    def test_outputs(self):
        # Set input point and label
        input_point = np.array([[500, 375]])
        input_label = np.array([1])

        # this function changes the input image order from (H, W, C) to (C, H, W)
        masks, scores, logits = self.processor.compute(self.image, input_point, input_label)

        # Check the outputs dimensions
        self.assertEqual(len(masks), 3)
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(logits), 3)

        # check the type of the outputs
        self.assertIsInstance(masks, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertIsInstance(logits, np.ndarray)

        # check the dtype of the outputs
        self.assertEqual(masks.dtype, np.float32)
        self.assertEqual(scores.dtype, np.float32)
        self.assertEqual(logits.dtype, np.float32)

        # check the shape of the outputs
        # the output has transposed shape compared to the input image
        # e.g. source is (C, H, W) and target is (H, W, C)
        self.assertEqual(masks.shape, self.image.transpose(2, 0, 1).shape)
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(logits.shape, (3, 256, 256))


if __name__ == "__main__":
    unittest.main()
