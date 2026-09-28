# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the inference module's actual identity model without TensorRT or a GPU."""

import argparse
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "v4l2_inference_holoviz" / "identity_model.onnx"
)


class IdentityModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = onnx.load(MODEL_PATH)

    def test_model_contract(self):
        onnx.checker.check_model(self.model)
        self.assertEqual(len(self.model.graph.input), 1)
        self.assertEqual(len(self.model.graph.output), 1)
        for binding, name in [
            (self.model.graph.input[0], "input"),
            (self.model.graph.output[0], "output"),
        ]:
            self.assertEqual(binding.name, name)
            self.assertEqual(binding.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
            self.assertEqual(
                [dim.dim_value for dim in binding.type.tensor_type.shape.dim],
                [1, 3, 256, 256],
            )
        self.assertEqual([node.op_type for node in self.model.graph.node], ["Identity"])

    def test_rgb_identity(self):
        pixels = np.random.default_rng(0).random((1, 3, 256, 256), dtype=np.float32)
        output = ReferenceEvaluator(self.model).run(["output"], {"input": pixels})[0]
        self.assertEqual(output.shape, pixels.shape)
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_array_equal(output, pixels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args, unittest_args = parser.parse_known_args()
    MODEL_PATH = args.model
    unittest.main(argv=[__file__, *unittest_args])
