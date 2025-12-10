"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes  # noqa

import cupy as cp  # noqa
import numpy as np
from cuda import cuda, nvrtc
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec, Tensor  # noqa
from holoscan.operators import HolovizOp


class CudaOperator(Operator):
    def __init__(
        self,
        fragment,
        *args,
        cuda_context=None,
        frame_spec=(800, 600, 3),  # (width, height, channels)
        create_stream=False,
        create_memory=False,
        **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self._cu_ctx = cuda_context
        self._frame_width, self._frame_height, self._frame_channels = frame_spec
        self._frame_shape = (self._frame_height, self._frame_width, self._frame_channels)

        # Initialize the CUDA context/stream/memory
        (err,) = cuda.cuInit(0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        (err,) = cuda.cuCtxSetCurrent(self._cu_ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS
        err, self._cu_device = cuda.cuCtxGetDevice()
        assert err == cuda.CUresult.CUDA_SUCCESS
        # Set the flag to indicate if the device is integrated or discrete
        err, self._is_integrated = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_INTEGRATED, self._cu_device
        )
        assert err == cuda.CUresult.CUDA_SUCCESS

        if create_stream:
            # `0` or `cuda.CUstream_flags.CU_STREAM_DEFAULT.value`      : default stream
            # `1` or `cuda.CUstream_flags.CU_STREAM_NON_BLOCKING.value` : non-blocking stream
            # Currently, there is no way to set a non-default CUDA stream to HolovizOp directly
            # (without using the CudaStreamPool resource). So, we are using the default stream here.
            err, self._stream = cuda.cuStreamCreate(0)
            assert err == cuda.CUresult.CUDA_SUCCESS
        else:
            self._stream = None

        if create_memory:
            self._frame_mem = self._allocate(self.get_byte_count())
        else:
            self._frame_mem = None

        self._module = None
        self._kernel = None

    def get_byte_count(self):
        return self._frame_height * self._frame_width * self._frame_channels

    def _allocate(self, size, flags=0):
        if self._is_integrated == 0:
            # This is a discrete device, so we can allocate using cuMemAlloc
            err, self._device_deviceptr = cuda.cuMemAlloc(size)
            assert err == cuda.CUresult.CUDA_SUCCESS
            return int(self._device_deviceptr)
        else:
            # This is an integrated device (e.g., Tegra), so we need to use cuMemHostAlloc
            err, self._host_deviceptr = cuda.cuMemHostAlloc(size, flags)
            assert err == cuda.CUresult.CUDA_SUCCESS
            err, device_deviceptr = cuda.cuMemHostGetDevicePointer(self._host_deviceptr, 0)
            assert err == cuda.CUresult.CUDA_SUCCESS
            return int(device_deviceptr)

    def _calculate_optimal_block_size(self, func):
        err, min_grid_size, optimal_block_size = cuda.cuOccupancyMaxPotentialBlockSize(
            func, None, 0, 0
        )
        assert err == cuda.CUresult.CUDA_SUCCESS

        return optimal_block_size, min_grid_size

    def _determine_block_dims(self, optimal_block_size):
        """Function to determine the 2D block size from the optimal block size."""
        block_dim = (1, 1, 1)
        while int(block_dim[0] * block_dim[1] * 2) <= optimal_block_size:
            if block_dim[0] > block_dim[1]:
                block_dim = (block_dim[0], block_dim[1] * 2, block_dim[2])
            else:
                block_dim = (block_dim[0] * 2, block_dim[1], block_dim[2])

        return block_dim

    def build_kernel(self, src_code):
        # Get the compute capability of the device
        err, major = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self._cu_device
        )
        assert err == cuda.CUresult.CUDA_SUCCESS
        err, minor = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self._cu_device
        )
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Compile program
        err, prog = nvrtc.nvrtcCreateProgram(str.encode(src_code), b"apply_gain.cu", 0, [], [])
        assert err == cuda.CUresult.CUDA_SUCCESS
        # opts = [b"--fmad=false", b"--gpu-architecture=compute_75"]
        opts = [b"--fmad=false", bytes("--gpu-architecture=sm_" + str(major) + str(minor), "ascii")]
        (err,) = nvrtc.nvrtcCompileProgram(prog, 2, opts)

        # Print log message if compilation fails
        if err != cuda.CUresult.CUDA_SUCCESS:
            err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
            assert err == cuda.CUresult.CUDA_SUCCESS
            log = b" " * log_size
            (err,) = nvrtc.nvrtcGetProgramLog(prog, log)
            assert err == cuda.CUresult.CUDA_SUCCESS
            result = log.decode()
            if len(result) > 1:
                print(result)
            raise Exception("Failed to compile the program")

        # Get PTX from compilation
        err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
        ptx = b" " * ptx_size
        (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Load PTX as module data and retrieve function
        ptx = np.char.array(ptx)
        # Note: Incompatible --gpu-architecture would be detected here
        err, self._module = cuda.cuModuleLoadData(ptx.ctypes.data)
        assert err == cuda.CUresult.CUDA_SUCCESS
        err, self._kernel = cuda.cuModuleGetFunction(self._module, b"apply_gain")
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Calculate the optimal block size for max occupancy
        optimal_block_size, min_grid_size = self._calculate_optimal_block_size(self._kernel)
        self.block_dims = self._determine_block_dims(optimal_block_size)
        self.grid_dims = (
            (self._frame_width + self.block_dims[0] - 1) // self.block_dims[0],
            (self._frame_height + self.block_dims[1] - 1) // self.block_dims[1],
            1,
        )
        if min_grid_size > self.grid_dims[0] * self.grid_dims[1]:
            # If the grid size is less than the minimum total grid size, adjust the grid size.
            self.grid_dims = (
                (min_grid_size + self.grid_dims[1]) // self.grid_dims[1],
                self.grid_dims[1],
                1,
            )

    def launch_kernel(self, args, stream=None):
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        (err,) = cuda.cuLaunchKernel(
            self._kernel,
            self.grid_dims[0],  # grid x dim
            self.grid_dims[1],  # grid y dim
            self.grid_dims[2],  # grid z dim
            self.block_dims[0],  # block x dim
            self.block_dims[1],  # block y dim
            self.block_dims[2],  # block z dim
            0,  # dynamic shared memory
            stream,  # stream
            args.ctypes.data,  # kernel arguments
            0,  # extra (ignore)
        )

    def stop(self):
        (err,) = cuda.cuCtxSetCurrent(self._cu_ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS

        if self._stream:
            (err,) = cuda.cuStreamSynchronize(self._stream)
            assert err == cuda.CUresult.CUDA_SUCCESS
            (err,) = cuda.cuStreamDestroy(self._stream)
            assert err == cuda.CUresult.CUDA_SUCCESS
            self._stream = None

        if self._frame_mem:
            if self._is_integrated == 0:
                (err,) = cuda.cuMemFree(self._device_deviceptr)
                self._device_deviceptr = None
            else:
                (err,) = cuda.cuMemFreeHost(self._host_deviceptr)
                self._host_deviceptr = None
            assert err == cuda.CUresult.CUDA_SUCCESS

        if self._kernel:
            (err,) = cuda.cuModuleUnload(self._module)
            assert err == cuda.CUresult.CUDA_SUCCESS
            self._kernel = None
            self._module = None

        # Call the parent stop method
        super().stop()


class CudaTxOp(CudaOperator):
    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Set the current context (because this operator may be executed in a different thread)
        (err,) = cuda.cuCtxSetCurrent(self._cu_ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Fill the memory with 1
        (err,) = cuda.cuMemsetD8(self._frame_mem, 1, self.get_byte_count())
        assert err == cuda.CUresult.CUDA_SUCCESS
        (err,) = cuda.cuStreamSynchronize(self._stream)
        assert err == cuda.CUresult.CUDA_SUCCESS

        d_x = np.array([int(self._frame_mem)], dtype=np.uint64)

        # Pass the array data with a CUDA stream to the output.
        # Note:
        #   `d_x` is a pointer to the device memory (`int(self._frame_mem)`. Type: `numpy.ndarray`).
        op_output.emit((d_x, self._stream), "out")


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
        value = op_input.receive("in")  # type: Tuple[numpy.ndarray, cuda.cuda.CUstream]

        # Set the current context (because this operator may be executed in a different thread)
        (err,) = cuda.cuCtxSetCurrent(self._cu_ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Destructure the value tuple
        value, stream = value

        # Adjust the multiplier based on the index
        multiplier = (self.index % 1000) / 1000.0 * self.multiplier
        self.index += 1

        # Call the kernel
        alpha = np.array([multiplier], dtype=np.float32)
        width = np.array(self._frame_width, dtype=np.uint64)
        height = np.array(self._frame_height, dtype=np.uint64)
        num_channels = np.array(self._frame_channels, dtype=np.uint32)

        args = [alpha, value, width, height, num_channels]
        self.launch_kernel(args, stream=stream)

        # Pass the array data with a CUDA stream to the output.
        op_output.emit((value, stream), "out")


class CudaRxOp(CudaOperator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")  # type: Tuple[numpy.ndarray, cuda.cuda.CUstream]

        # Set the current context (because this operator may be executed in a different thread)
        (err,) = cuda.cuCtxSetCurrent(self._cu_ctx)
        assert err == cuda.CUresult.CUDA_SUCCESS

        # Destructure the value tuple
        value, stream = value

        ###############################################################
        # Example 1: Creating numpy array from the cuda/cudahost memory (12.011s)
        ###############################################################

        # h_x = np.zeros(self._frame_shape).astype(dtype=np.uint8)
        # (err,) = cuda.cuMemcpyDtoHAsync(h_x.ctypes.data, value, self.get_byte_count(), stream)
        # assert err == cuda.CUresult.CUDA_SUCCESS
        # (err,) = cuda.cuStreamSynchronize(stream)
        # assert err == cuda.CUresult.CUDA_SUCCESS

        # op_output.emit({"": h_x}, "out")
        # return
        # # # or, you can create a numpy array from any pointer (e.g., `h_x.ctypes.data`) as shown below
        # # uint8_pointer_type = ctypes.POINTER(ctypes.c_uint8)
        # # numpy_array = np.ctypeslib.as_array(
        # #     ctypes.cast(h_x.ctypes.data, uint8_pointer_type),
        # #     self._frame_shape,
        # # )
        # # op_output.emit({"": numpy_array}, "out")
        # # return

        ###############################################################
        # Example 2: Converting NumPy array to CuPy array (11.729s)
        #
        # This might be slightly faster than Example 1 because CuPy uses an internal GPU memory pool
        # for copying data from the CPU to the GPU. Otherwise, the visualizer will copy data from
        # the CPU to GPU memory for rendering (slow path).
        #
        # Note: Install CuPy with the following command
        #
        #         python -m pip install cupy-cuda12x
        ###############################################################

        # h_x = np.zeros((self._frame_height, self._frame_width, self._frame_channels)).astype(
        #     dtype=np.uint8
        # )
        # (err,) = cuda.cuMemcpyDtoHAsync(h_x.ctypes.data, value, self.get_byte_count(), stream)
        # (err,) = cuda.cuStreamSynchronize(stream)

        # uint8_pointer_type = ctypes.POINTER(ctypes.c_uint8)

        # numpy_array = np.ctypeslib.as_array(
        #     ctypes.cast(h_x.ctypes.data, uint8_pointer_type),
        #     (self._frame_height, self._frame_width, self._frame_channels),
        # )

        # cupy_array = cp.asarray(numpy_array)

        # op_output.emit({"": cupy_array}, "out")
        # return

        ###############################################################
        # Example 3: Creating object having array interface from the cuda/cudahost memory (3.504s)
        ###############################################################

        class CudaArray:
            """Class to represent a CUDA array interface object."""

            pass

        cuda_array = CudaArray()

        # Reference: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
        cuda_array.__cuda_array_interface__ = {
            "shape": (self._frame_height, self._frame_width, self._frame_channels),
            "typestr": np.dtype(np.uint8).str,  # "|u1"
            "descr": [("", np.dtype(np.uint8).str)],
            "stream": int(stream),
            "version": 3,
            "strides": None,
            "data": (int(value), False),
        }

        op_output.emit({"": cuda_array}, "out")
        return
        # # This is same with the following code
        # op_output.emit({"": Tensor.as_tensor(cuda_array)}, "out")
        # return

        ###############################################################
        # Example 4: Creating cupy array from the cuda/cudahost memory (3.449s)
        #
        # Note: Install CuPy with the following command
        #
        #         python -m pip install cupy-cuda12x
        ###############################################################

        # cupy_array = cp.ndarray(
        #     (self._frame_height, self._frame_width, self._frame_channels),
        #     dtype=cp.uint8,
        #     memptr=cp.cuda.MemoryPointer(
        #         cp.cuda.UnownedMemory(int(value), self.get_byte_count(), owner=self, device_id=0),
        #         0,
        #     ),
        # )

        # op_output.emit({"": cupy_array}, "out")
        # return


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
            # print(f"  dtype: {tensor.dtype}")
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
        (err,) = cuda.cuInit(0)
        assert err == cuda.CUresult.CUDA_SUCCESS
        cu_device_ordinal = 0
        err, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
        assert err == cuda.CUresult.CUDA_SUCCESS
        err, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
        assert err == cuda.CUresult.CUDA_SUCCESS
        self.cu_device = cu_device

        width = 800
        height = 600
        channels = 3
        multiplier = 200

        # Define the tx, mx, rx operators, allowing the tx operator to execute 10000 times
        tx = CudaTxOp(
            self,
            CountCondition(self, 10000),
            name="tx",
            cuda_context=cu_context,
            frame_spec=(width, height, channels),
            create_stream=True,
            create_memory=True,
        )
        mx = ApplyGainOp(
            self,
            name="mx",
            cuda_context=cu_context,
            multiplier=multiplier,
            frame_spec=(width, height, channels),
        )
        rx = CudaRxOp(
            self,
            name="rx",
            cuda_context=cu_context,
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

    def __del__(self):
        if hasattr(super(), "__del__"):
            super().__del__()

        (err,) = cuda.cuDevicePrimaryCtxRelease(self.cu_device)
        assert err == cuda.CUresult.CUDA_SUCCESS


def main():
    app = TestCudaApp()
    app.run()


if __name__ == "__main__":
    main()
