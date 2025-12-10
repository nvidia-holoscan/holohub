"""
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""  # noqa: E501

import cupy as cp  # noqa
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec, Tensor  # noqa
from holoscan.operators import HolovizOp


class CudaOperator(Operator):
    BLOCK_SIZE = 32  # CUDA block size

    def __init__(
        self,
        fragment,
        *args,
        frame_spec=(800, 600, 3),  # (width, height, channels)
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self._frame_width, self._frame_height, self._frame_channels = frame_spec
        self._frame_shape = (self._frame_height, self._frame_width, self._frame_channels)

        self._kernel = None

    def build_kernel(self, src_code, method_name="apply_gain"):
        if self._kernel is None:
            # https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html
            self._kernel = cp.RawKernel(src_code, method_name, options=("--fmad=false",))

            # Calculate the grid and block dimensions
            self.block_dims = (self.BLOCK_SIZE, self.BLOCK_SIZE, 1)
            self.grid_dims = (
                (self._frame_width + self.block_dims[0] - 1) // self.block_dims[0],
                (self._frame_height + self.block_dims[1] - 1) // self.block_dims[1],
                1,
            )

    def launch_kernel(self, args):
        self._kernel(
            self.grid_dims,
            self.block_dims,
            args,
        )


class ApplyGainOp(CudaOperator):
    def __init__(self, fragment, *args, multiplier=2.0, **kwargs):
        self.multiplier = multiplier
        self.index = 0

        super().__init__(fragment, *args, **kwargs)

        src_code = r"""
            extern "C" __global__
            void apply_gain(float alpha, unsigned char *image, size_t width, size_t height, int num_channels)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= width || y >= height)
                    return;

                const int thread_id = y * width + x;
                const float grad = static_cast<float>(thread_id) / (width * height);

                const int index = thread_id * num_channels;

                for (int i = 0; i < num_channels; i++) {
                    const float value = (i == 1) ? alpha * image[index + i] : alpha * grad * image[index + i];
                    image[index + i] = fminf(value, 255.0f);
                }
            }
            """

        self.build_kernel(src_code)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")  # type: cupy.ndarray

        # Adjust the multiplier based on the index
        multiplier = (self.index % 1000) / 1000.0 * self.multiplier
        self.index += 1

        # Call the kernel
        alpha = cp.float32(multiplier)
        width = cp.uint64(self._frame_width)
        height = cp.uint64(self._frame_height)
        num_channels = self._frame_channels

        args = (alpha, value, width, height, num_channels)

        self.launch_kernel(args)

        # This took about 3.674s which is slightly slower than the CUDA Python version

        # cp.cuda.Stream.null.synchronize()  # not doing a redundant sync here for efficiency
        op_output.emit(value, "out")


class CudaTxOp(CudaOperator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

        # Pre-allocate the array
        self.cp_array = cp.empty(self._frame_shape, dtype=cp.uint8)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Set the array to 1 for initialization
        self.cp_array.fill(1)
        op_output.emit(self.cp_array, "out")


class CudaRxOp(CudaOperator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        # Emit as a dictionary (dict[str, holoscan.core.Tensor])
        op_output.emit({"": value}, "out")


class ProbeOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")  # type: dict[str, holoscan.core.Tensor]

        for key, tensor in value.items():
            if hasattr(tensor, "__cuda_array_interface__"):
                array_interface = tensor.__cuda_array_interface__
                # print("#tensor.__cuda_array_interface__", tensor.__cuda_array_interface__)
            if hasattr(tensor, "__array_interface__"):
                # print("#tensor.__array_interface__", tensor.__array_interface__)
                array_interface = tensor.__array_interface__  # noqa: F841
            # print(f"Tensor name: {key}")
            # print(f"  shape: {tensor.shape}")
            # print(f"  dtype: {tensor.dtype}")7.445s
            # print(f"  is_contiguous: {tensor.is_contiguous()}")
            # print(f"  strides: {tensor.strides}")
            # print(f"  device: {tensor.device}")
            # print(f"  nbytes: {tensor.nbytes}")
            # print(f"  size: {tensor.size}")
            # print(f"  ndim: {tensor.ndim}")
            # print(f"  itemsize: {tensor.itemsize}")
            # # Since v2.1, tensor.data returns `int` value. Otherwise, use `array_interface['data'][0]` to get the int value
            # print(f"  data: {tensor.data} == {array_interface['data'][0]}")


class TestCudaApp(Application):
    def compose(self):
        width = 800
        height = 600
        channels = 3
        multiplier = 200

        # Define the tx, mx, rx operators, allowing the tx operator to execute 10000 times
        tx = CudaTxOp(
            self,
            CountCondition(self, 10000),
            name="tx",
            frame_spec=(width, height, channels),
        )
        mx = ApplyGainOp(
            self,
            name="mx",
            multiplier=multiplier,
            frame_spec=(width, height, channels),
        )
        rx = CudaRxOp(
            self,
            name="rx",
            frame_spec=(width, height, channels),
        )

        # Define the workflow: tx -> mx -> rx
        self.add_flow(tx, mx)
        self.add_flow(mx, rx)

        probe = ProbeOp(self, name="probe")

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            tensors=[
                # `name=""` here to match the output of VideoStreamReplayerOp
                dict(name="", type="color", opacity=1.0, priority=0),
            ],
        )

        # -> rx -> probe
        #     └─-> visualizer
        self.add_flow(rx, probe)
        self.add_flow(rx, visualizer, {("out", "receivers")})


def main():
    app = TestCudaApp()
    app.run()


if __name__ == "__main__":
    main()
