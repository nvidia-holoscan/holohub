# Best Practices to integrate external libraries into Holoscan pipelines

The Holoscan SDK is part of NVIDIA Holoscan, the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including medical devices, high-performance computing at the edge, industrial inspection, and more.

With the Holoscan SDK, one can develop an end-to-end GPU-accelerated pipeline with RDMA support. However, with increasing requirements for pre-processing and post-processing beyond inference-only pipelines, integration with other powerful, GPU-accelerated libraries is needed.

<div align="center">
<img src="./images/typical_pipeline_holoscan.png" style="border: 2px solid black;">
</div>

One of the key features of the Holoscan SDK is its seamless interoperability with other libraries.

This tutorial explains how to leverage this capability in your applications.
For detailed examples of integrating various libraries with Holoscan applications, refer to the following sections:
- Tensor Interoperability
  - [Integrate **MatX** library](#integrate-matx-library) - DLPack support in C++
  - [Integrate RAPIDS **cuCIM** library](#integrate-rapids-cucim-library)
  - [Integrate **CV-CUDA** library](#integrate-cv-cuda-library)
  - [Integrate **OpenCV with CUDA Module**](#integrate-opencv-with-cuda-module)
  - [Integrate **PyTorch** library](#integrate-pytorch-library)
- CUDA Interoperability
  - [Integrate **CUDA Python** library](#integrate-cuda-python-library)
  - [Integrate **CuPy** library](#integrate-cupy-library)

## Interoperability Features

### DLPack Support

The Holoscan SDK supports [DLPack](https://dmlc.github.io/dlpack/latest/), enabling efficient data exchange between deep learning frameworks.

### Array Interface Support

The SDK also supports the array interface, including:
- [`__array_interface__`](https://numpy.org/doc/stable/reference/arrays.interface.html)
- [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)

This allows for seamless integration with various Python libraries such as:
- [CuPy](https://docs.cupy.dev/en/stable/user_guide/interoperability.html)
- [PyTorch](https://github.com/pytorch/pytorch/issues/15601)
- [JAX](https://github.com/google/jax/issues/1100#issuecomment-580773098)
- [TensorFlow](https://github.com/tensorflow/community/pull/180)
- [Numba](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html)

### Technical Details

The `Tensor` class is a wrapper around the `DLManagedTensorContext` struct, which holds the `DLManagedTensor` object (a [DLPack structure](https://dmlc.github.io/dlpack/latest/c_api.html#c.DLManagedTensor)).

For more information on interoperability, refer to the following sections in the Holoscan SDK documentation:
- [Interoperability between GXF and native C++ operators](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#interoperability-between-gxf-and-native-c-operators)
- [Interoperability between wrapped and native Python operators](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#interoperability-between-wrapped-and-native-python-operators)

### CUDA Array Interface Support

The following Python libraries have adopted the [CUDA Array Interface](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html#interoperability):

- [CuPy](https://docs-cupy.chainer.org/en/stable/reference/interoperability.html)
- [CV-CUDA](https://github.com/CVCUDA/CV-CUDA)
- [PyTorch](https://pytorch.org/)
- [Numba](https://numba.readthedocs.io/en/stable/user/5minguide.html)
- [PyArrow](https://arrow.apache.org/docs/python/generated/pyarrow.cuda.Context.html#pyarrow.cuda.Context.buffer_from_object)
- [mpi4py](https://mpi4py.readthedocs.io/en/latest/overview.html#support-for-cuda-aware-mpi)
- [ArrayViews](https://github.com/xnd-project/arrayviews)
- [JAX](https://jax.readthedocs.io/en/latest/index.html)
- [PyCUDA](https://documen.tician.de/pycuda/tutorial.html#interoperability-with-other-libraries-using-the-cuda-array-interface)
- [DALI: the NVIDIA Data Loading Library](https://github.com/NVIDIA/DALI) :
  - [TensorGPU objects](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/data_types.html#nvidia.dali.backend.TensorGPU) expose the CUDA Array Interface.
  - [The External Source operator](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.fn.external_source) consumes objects exporting the CUDA Array Interface.
- The RAPIDS stack:
  - [cuCIM](https://github.com/rapidsai/cucim)
  - [cuDF](https://docs.rapids.ai/api/cudf/stable/user_guide/10min/)
  - [cuML](https://docs.rapids.ai/api/cuml/nightly/)
  - [cuSignal](https://github.com/rapidsai/cusignal)
  - [RMM](https://docs.rapids.ai/api/rmm/stable/guide/)

For more details on using the CUDA Array Interface and DLPack with various libraries, see [CuPy's Interoperability guide](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#).

### Using Holoscan Tensors in Python

The Holoscan SDK's Python API provides the `holoscan.as_tensor()` method to convert objects supporting the (CUDA) Array Interface or DLPack to a Holoscan Tensor. The `holoscan.Tensor` object itself also supports these interfaces, allowing for easy integration with compatible libraries.

Example usage:

```python
import cupy as cp
import numpy as np
import torch
import holoscan as hs

# Create tensors using different libraries
torch_cpu_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch_gpu_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
numpy_tensor = np.array([[1, 2, 3], [4, 5, 6]])
cupy_tensor = cp.array([[1, 2, 3], [4, 5, 6]])

# Convert to Holoscan Tensors
torch_cpu_to_holoscan = hs.as_tensor(torch_cpu_tensor)
torch_gpu_to_holoscan = hs.as_tensor(torch_gpu_tensor)
numpy_to_holoscan = hs.as_tensor(numpy_tensor)
cupy_to_holoscan = hs.as_tensor(cupy_tensor)
```

## Tensor Interoperability

### Integrate MatX library

[MatX library (An efficient C++17 GPU numerical computing library with Python-like syntax)](https://github.com/NVIDIA/MatX) is an open-source, efficient C++17 GPU numerical computing library created by NVIDIA. It provides a NumPy-like interface for GPU-accelerated numerical computing, enabling developers to write high-performance, GPU-accelerated code with ease.

#### Installation

MatX is a header-only library. Using it in your own projects is as simple as including only the core `matx.h` file.

Please refer to the [MatX documentation](https://nvidia.github.io/MatX/build.html) for detailed instructions on building and using the MatX library.

The following is a sample CMakeLists.txt file for a project that uses MatX:

```
cmake_minimum_required(VERSION 3.20)
project(my_app CXX)

# Holoscan
find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
set(CMAKE_CUDA_ARCHITECTURES "70;80")
enable_language(CUDA)

# Uncomment below for debug build
# set(CMAKE_BUILD_TYPE Debug)

# Download MatX
# OBS: Currently pulls a specific commit of MatX (with host support for find_idx)
include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG main
)
FetchContent_MakeAvailable(MatX)

add_executable(my_app
  my_app.cpp
)

target_link_libraries(my_app
  PRIVATE
  holoscan::core
  holoscan::ops::aja
  holoscan::ops::video_stream_replayer
  holoscan::ops::format_converter
  holoscan::ops::inference
  holoscan::ops::segmentation_postprocessor
  holoscan::ops::holoviz
  matx::matx
)

```

#### Sample code

The following are the sample applications that use MatX library to integrate with Holoscan SDK.

- [Multi AI Application with SSD Detection and MONAI Endoscopic Tool Segmentation](https://github.com/nvidia-holoscan/holohub/tree/main/applications/multiai_endoscopy)
  - `applications/multiai_endoscopy`
- [Network Radar Pipeline](https://github.com/nvidia-holoscan/holohub/tree/main/applications/network_radar_pipeline/cpp)
  - `applications/network_radar_pipeline/cpp`
- [Simple Radar Pipeline Application](https://github.com/nvidia-holoscan/holohub/tree/main/applications/simple_radar_pipeline/cpp)
  - `applications/simple_radar_pipeline/cpp`

**On the GPU**

- <https://github.com/nvidia-holoscan/holohub/blob/main/applications/multiai_endoscopy/cpp/post-proc-matx-gpu/multi_ai.cu>

```cpp
#include <holoscan/holoscan.hpp>
#include <matx.h>

// ...

void compute(InputContext& op_input, OutputContext& op_output,
             ExecutionContext& context) override {
  // Get input message and make output message
  auto in_message = op_input.receive<gxf::Entity>("in").value();
  // ...
  auto boxes = in_message.get<Tensor>("inference_output_detection_boxes");
  auto scores = in_message.get<Tensor>("inference_output_detection_scores");
  int32_t Nb = scores->shape()[1];  // Number of boxes
  // ...
  auto boxesl_mx = matx::make_tensor<float>({1, Nl(), 4});
  (boxesl_mx = matx::remap<1>(boxes_ix_mx, ixl_mx)).run();
  // ...
  // Holoscan tensors to MatX tensors
  auto boxes_mx = matx::make_tensor<float>((float*)boxes->data(), {1, Nb, 4});
  // ...
  // MatX to Holoscan tensor
  auto boxes_hs = std::make_shared<holoscan::Tensor>(boxesls_mx.GetDLPackTensor());
  // ...
}
```

**On the CPU**

- <https://github.com/nvidia-holoscan/holohub/blob/main/applications/multiai_endoscopy/cpp/post-proc-matx-cpu/multi_ai.cpp>

MatX library usage on the CPU is similar to the GPU version, but the `run()` function is called with `matx::SingleThreadHostExecutor()` to run the operation on the CPU.

```cpp
#include <holoscan/holoscan.hpp>
#include <matx.h>

// ...

void compute(InputContext& op_input, OutputContext& op_output,
             ExecutionContext& context) override {
  // Get input message and make output message
  auto in_message = op_input.receive<gxf::Entity>("in").value();
  // ...
  auto boxesh = in_message.get<Tensor>("inference_output_detection_boxes");  // (1, num_boxes, 4)
  auto scoresh = in_message.get<Tensor>("inference_output_detection_scores");  // (1, num_boxes)
  int32_t Nb = scoresh->shape()[1];  // Number of boxes
  // ...
  auto boxes = copy_device2vec<float>(boxesh);
  // Holoscan tensors to MatX tensors
  auto boxes_mx = matx::make_tensor<float>(boxes.data(), {1, Nb, 4});
  // ...
  auto boxesl_mx = matx::make_tensor<float>({1, Nl(), 4});
  (boxesl_mx = matx::remap<1>(boxes_ix_mx, ixl_mx)).run(matx::SingleThreadHostExecutor());
  // ...
  // MatX to Holoscan tensor
  auto boxes_hs = std::make_shared<holoscan::Tensor>(boxesls_mx.GetDLPackTensor());
  // ...
}
```

### Integrate RAPIDS cuCIM library

[RAPIDS cuCIM](https://github.com/rapidsai/cucim) (Compute Unified Device Architecture Clara IMage) is an open-source, accelerated computer vision and image processing software library for multidimensional images used in biomedical, geospatial, material and life science, and remote sensing use cases.

See the supported Operators in [cuCIM documentation](https://docs.rapids.ai/api/cucim/stable/).

cuCIM offers interoperability with CuPy. We can initialize CuPy arrays directly from Holoscan Tensors and use the arrays in cuCIM operators for processing without memory transfer between host and device.

#### Installation

Follow the [cuCIM documentation](https://github.com/rapidsai/cucim?tab=readme-ov-file#install-cucim) to install the RAPIDS cuCIM library.

#### Sample code

Sample code as below:

```py
import cupy as cp
import cucim.skimage.exposure as cu_exposure
from cucim.skimage.util import img_as_ubyte
from cucim.skimage.util import img_as_float

def CustomizedcuCIMOperator(Operator):
    ### Other implementation of __init__, setup()... etc.

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input_tensor")
        input_tensor = message.get()
        # Directly use Holoscan tensor to initialize CuPy array
        cp_array = cp.asarray(input_tensor)

        cp_array = img_as_float(cp_array)
        cp_res=cu_exposure.equalize_adapthist(cp_array)
        cp_array = img_as_ubyte(cp_res)

        # Emit CuPy array memory as an item in a `holoscan.TensorMap`
        op_output.emit(dict(out_tensor=cp_array), "out")

```

### Integrate CV-CUDA library

[CV-CUDA](https://github.com/CVCUDA/CV-CUDA) is an open-source, graphics processing unit (GPU)-accelerated library for cloud-scale image processing and computer vision developed jointly by NVIDIA and the ByteDance Applied Machine Learning teams. CV-CUDA helps developers build highly efficient pre- and post-processing pipelines that can improve throughput by more than 10x while lowering cloud computing costs.

See the supported CV-CUDA Operators in the [CV-CUDA developer guide](https://github.com/CVCUDA/CV-CUDA/blob/main/DEVELOPER_GUIDE.md)

#### Installation

Follow the [CV-CUDA documentation](https://cvcuda.github.io/installation.html) to install the CV-CUDA library.

Requirement: CV-CUDA >= 0.2.1 (From which version DLPack interop is supported)

#### Sample code

CV-CUDA is implemented with DLPack standards. So, CV-CUDA tensor can directly access Holocan Tensor.

Refer to the [Holoscan CV-CUDA sample application](https://github.com/nvidia-holoscan/holohub/tree/main/applications/cvcuda_basic) for an example of how to use CV-CUDA with Holoscan SDK.

```py
import cvcuda

class CustomizedCVCUDAOp(Operator):
    def __init__(self, *args, **kwargs):

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input_tensor")
        spec.output("output_tensor")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input_tensor")
        input_tensor = message.get()

        cvcuda_input_tensor = cvcuda.as_tensor(input_tensor,"HWC")

        cvcuda_resize_tensor = cvcuda.resize(
                    cvcuda_input_tensor,
                    (
                        640,
                        640,
                        3,
                    ),
                    cvcuda.Interp.LINEAR,
                )

        buffer = cvcuda_resize_tensor.cuda()

        # Emits an `holoscan.TensorMap` with a single entry `out_tensor`
        op_output.emit(dict(out_tensor=buffer), "output_tensor")

```

### Integrate OpenCV with CUDA Module

[OpenCV](https://opencv.org/) (Open Source Computer Vision Library) is a comprehensive open-source library that contains over 2500 algorithms covering Image & Video Manipulation, Object and Face Detection, OpenCV Deep Learning Module and much more.

[OpenCV also supports GPU [acceleration](https://docs.opencv.org/4.8.0/d2/dbc/cuda_intro.html) and includes a CUDA module which is a set of classes and functions to utilize CUDA computational capabilities. It is implemented using NVIDIA CUDA Runtime API and provides utility functions, low-level vision primitives, and high-level algorithms.

#### Installation

Prerequisites:
- OpenCV >= 4.8.0 (From which version, OpenCV GpuMat supports initialization with GPU Memory pointer)

Install OpenCV with its CUDA module following the guide in [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib/tree/4.x)

We also recommend referring to the [Holoscan Endoscopy Depth Estimation application container](https://github.com/nvidia-holoscan/holohub/blob/main/applications/endoscopy_depth_estimation/Dockerfile) as an example of how to build an image with Holoscan SDK and OpenCV CUDA.

#### Sample code

The data type of OpenCV is GpuMat which implements neither the __cuda_array_interface__ nor the standard DLPack. To achieve the end-to-end GPU accelerated pipeline/application, we need to implement 2 functions to convert the GpuMat to CuPy array which can be accessed directly with Holoscan Tensor and vice versa.

Refer to the [Holoscan Endoscopy Depth Estimation sample application](https://github.com/nvidia-holoscan/holohub/tree/main/applications/cvcuda_basic) for an example of how to use the OpenCV operator with Holoscan SDK.

1. Conversion from GpuMat to CuPy Array

The GpuMat object of OpenCV Python bindings provides a cudaPtr method that can be used to access the GPU memory address of a GpuMat object. This memory pointer can be utilized to initialize a CuPy array directly, allowing for efficient data handling by avoiding unnecessary data transfers between the host and device.

Refer to the function below, which is used to create a CuPy array from a GpuMat. For more details, see the source code in [holohub/applications/endoscopy_depth_estimation-gpumat_to_cupy](https://github.com/nvidia-holoscan/holohub/blob/main/applications/endoscopy_depth_estimation/endoscopy_depth_estimation.py#L52).

```py
import cv2
import cupy as cp

def gpumat_to_cupy(gpu_mat: cv2.cuda.GpuMat) -> cp.ndarray:
    w, h = gpu_mat.size()
    size_in_bytes = gpu_mat.step * w
    shapes = (h, w, gpu_mat.channels())
    assert gpu_mat.channels() <=3, "Unsupported GpuMat channels"

    dtype = None
    if gpu_mat.type() in [cv2.CV_8U,cv2.CV_8UC1,cv2.CV_8UC2,cv2.CV_8UC3]:
        dtype = cp.uint8
    elif gpu_mat.type() == cv2.CV_8S:
        dtype = cp.int8
    elif gpu_mat.type() == cv2.CV_16U:
        dtype = cp.uint16
    elif gpu_mat.type() == cv2.CV_16S:
        dtype = cp.int16
    elif gpu_mat.type() == cv2.CV_32S:
        dtype = cp.int32
    elif gpu_mat.type() == cv2.CV_32F:
        dtype = cp.float32
    elif gpu_mat.type() == cv2.CV_64F:
        dtype = cp.float64

    assert dtype is not None, "Unsupported GpuMat type"

    mem = cp.cuda.UnownedMemory(gpu_mat.cudaPtr(), size_in_bytes, owner=gpu_mat)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    cp_out = cp.ndarray(
        shapes,
        dtype=dtype,
        memptr=memptr,
        strides=(gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()),
    )
    return cp_out

```

Note: In this function, we used the [UnownedMemory](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.UnownedMemory.html#cupy.cuda.UnownedMemory) API to create the CuPy array. In some cases, the OpenCV operators will allocate new device memory which needs to be handled by CuPy and the lifetime is not limited to one operator but the whole pipeline. In this case, the CuPy array initiated from the GpuMat shall know the owner and keep the reference to the object. Check the CuPy documentation for more details on [CuPy interoperability](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#device-memory-pointers).

2. Conversion from Holoscan Tensor to GpuMat via CuPy array

With the release of OpenCV 4.8, the Python bindings for OpenCV now support the initialization of GpuMat objects directly from GPU memory pointers. This capability facilitates more efficient data handling and processing by allowing direct interaction with GPU-resident data, bypassing the need for data transfer between host and device memory.

Within pipeline applications based on Holoscan SDK, the GPU Memory pointer can be obtained through the `__cuda_array_interface__` interface provided by CuPy arrays.

Refer to the functions outlined below for creating GpuMat objects utilizing CuPy arrays. For a detailed implementation, see the source code provided in [holohub/applications/endoscopy_depth_estimation-gpumat_from_cp_array](https://github.com/nvidia-holoscan/holohub/blob/main/applications/endoscopy_depth_estimation/endoscopy_depth_estimation.py#L28).

```py
import cv2
import cupy as cp
import holoscan as hs
from holoscan.gxf import Entity

def gpumat_from_cp_array(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    assert len(arr.shape) in (2, 3), "CuPy array must have 2 or 3 dimensions to be a valid GpuMat"
    type_map = {
        cp.dtype('uint8'): cv2.CV_8U,
        cp.dtype('int8'): cv2.CV_8S,
        cp.dtype('uint16'): cv2.CV_16U,
        cp.dtype('int16'): cv2.CV_16S,
        cp.dtype('int32'): cv2.CV_32S,
        cp.dtype('float32'): cv2.CV_32F,
        cp.dtype('float64'): cv2.CV_64F
    }
    depth = type_map.get(arr.dtype)
    assert depth is not None, "Unsupported CuPy array dtype"
    channels = 1 if len(arr.shape) == 2 else arr.shape[2]
    mat_type = depth + ((channels - 1) << 3)

     mat = cv2.cuda.createGpuMatFromCudaMemory(
      arr.__cuda_array_interface__['shape'][1::-1],
      mat_type,
      arr.__cuda_array_interface__['data'][0]
  )
    return mat
```

3. Integrate OpenCV Operators inside customized Operator

The demonstration code is provided below. For the complete source code, please refer to the [holohub/applications/endoscopy_depth_estimation-customized Operator](https://github.com/nvidia-holoscan/holohub/blob/main/applications/endoscopy_depth_estimation/endoscopy_depth_estimation.py#L161).

```py
   def compute(self, op_input, op_output, context):
        stream = cv2.cuda_Stream()
        message = op_input.receive("in")

        cp_frame = cp.asarray(message.get(""))  # CuPy array
        cv_frame = gpumat_from_cp_array(cp_frame)  # GPU OpenCV mat

        ## Call OpenCV Operator
        cv_frame = cv2.cuda.XXX(hsv_merge, cv2.COLOR_HSV2RGB)

        cp_frame = gpumat_to_cupy(cv_frame)
        cp_frame = cp.ascontiguousarray(cp_frame)

        op_output.emit(dict(out_tensor=cp_frame), "out")
```

### Integrate PyTorch library

[PyTorch](https://pytorch.org/) is a popular open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and dynamic computational graph that allows for easy model building and training. PyTorch also supports GPU acceleration, making it ideal for deep learning applications that require high-performance computing.

Since PyTorch tensors support the array interface and DLPack ([link](https://github.com/pytorch/pytorch/issues/15601)), they can be interoperable with other array/tensor objects including Holoscan Tensors.

#### Installation

Follow the [PyTorch documentation](https://pytorch.org/) to install the PyTorch library.

e.g., for CUDA 12.x with pip:

```bash
python3 -m pip install torch torchvision torchaudio
```

#### Sample code

The following is a sample application that demonstrates how to use PyTorch with Holoscan SDK:

```py
import torch

def CustomizedTorchOperator(Operator):
    ### Other implementation of __init__, setup()... etc.

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input_tensor")
        input_tensor = message.get()
        # Directly use Holoscan tensor to initialize PyTorch tensor
        torch_tensor = torch.as_tensor(input_tensor, device="cuda")

        torch_tensor *= 2

        # Emit PyTorch tensor memory as an item in a `holoscan.TensorMap`
        op_output.emit(dict(out_tensor=torch_tensor), "out")
```

## CUDA Interoperability

### Integrate CUDA Python library

[CUDA Python](https://developer.nvidia.com/cuda-python) is a Python library that provides Cython/Python wrappers for CUDA driver and runtime APIs. It offers a convenient way to leverage GPU acceleration for complex computations, making it ideal for high-performance applications that require intensive numerical processing.

When using CUDA Python with the Holoscan SDK, you need to use the Primary context ([CUDA doc link](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER.html#group__CUDART__DRIVER)) by calling `cuda.cuDevicePrimaryCtxRetain()` ([link](https://nvidia.github.io/cuda-python/module/cuda.html#primary-context-management)).

Since the Holoscan Operator is executed in an arbitrary non-main thread, you may need to set the CUDA context using the [cuda.cuCtxSetCurrent()](https://nvidia.github.io/cuda-python/module/cuda.html#cuda.cuda.cuCtxSetCurrent) method in the `Operator.compute()` method.

#### Installation

Follow the instructions in the [CUDA Python documentation](https://nvidia.github.io/cuda-python/install.html) to install the CUDA Python library.

CUDA Python can be installed using `pip`:

```bash
python3 -m pip install cuda-python
```

#### Sample code

The following is a sample application ([cuda_example.py](cuda_example.py)) that demonstrates how to use CUDA Python with Holoscan SDK:


<details>
<summary>cuda_example.py</summary>

```python
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
                array_interface = tensor.__array_interface__
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
```

</details>

In this example, we define a `CudaOperator` class that encapsulates the CUDA context, stream, and memory management. The `CudaOperator` class provides methods for allocating device memory, building CUDA kernels, launching kernels, and cleaning up. We also define three operators: `CudaTxOp`, `ApplyGainOp`, and `CudaRxOp`, which perform data initialization, apply a gain operation, and process the output data, respectively. The output of the `CudaRxOp` operator is passed to both a `ProbeOp` operator, which inspects the data and prints the metadata information, and a `HolovizOp` operator, which visualizes the data using the Holoviz module.

There are four examples in the `CudaRxOp.compute()` method that demonstrate different ways to handle data conversion and transfer between tensor libraries. These examples include creating 1) a NumPy array from CUDA memory, 2) converting a NumPy array to a CuPy array, 3) creating a CUDA array interface object, and 4) creating a CuPy array from CUDA memory.

`__cuda_array_interface__` is a dictionary that provides a standard interface for exchanging array data between different libraries in Python. It contains metadata such as the shape, data type, and memory location of the array. By using this interface, you can efficiently transfer tensor data between two libraries without copying the data.

In the following example, we create a `CudaArray` class to represent a CUDA array interface object and populate it with the necessary metadata. This object can then be passed to the `op_output.emit()` method to transfer the data to downstream operators.

In the `__cuda_array_interface__` dictionary, the `stream` field is the CUDA stream associated with the data. When passing a `cuda.cuda.CUstream` object (the variable named `stream`) to the `stream` field, you need to convert it to an integer using `int(stream)`:

```python
class CudaRxOp(CudaOperator):
    # ...
    def compute(self, op_input, op_output, context):
        # ...
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
        # ...
```

Please don't confuse this with the `cuda.cuda.CUstream.getPtr()` method. If you use the `stream.getPtr()` method, it will return a pointer to the CUDA stream object, not the stream ID. To get the stream ID, you need to convert the `stream` object to an integer using `int(stream)`. Otherwise, you will get an error that is difficult to debug, like this:

```bash
[error] [tensor.cpp:479] Runtime call "Failure during call to cudaEventRecord" in line 479 of file ../python/holoscan/core/tensor.cpp failed with 'context is destroyed' (709)
[error] [gxf_wrapper.cpp:84] Exception occurred for operator: 'rx' - RuntimeError: Error occurred in CUDA runtime API call
```

### Integrate CuPy library

[CuPy](https://cupy.dev/) is an open-source array library for GPU-accelerated computing with a NumPy-compatible API. It provides a convenient way to perform high-performance numerical computations on NVIDIA GPUs, making it ideal for applications that require efficient data processing and manipulation.

#### Installation

CuPy can be installed using `pip`:

```bash
python3 -m pip install cupy-cuda12x  # for CUDA v12.x
```

For more detailed installation instructions, refer to the [CuPy documentation](https://docs.cupy.dev/en/stable/install.html).

#### Sample code

The following is a sample application([cupy_example.py](cupy_example.py)) that demonstrates how to use CuPy with Holoscan SDK:

<details>
<summary>cupy_example.py</summary>

```python
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
                array_interface = tensor.__array_interface__
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
```

</details>

This example performs the same operations as the previous example but uses CuPy instead of [CUDA Python](#integrate-cuda-python-library). The `CudaOperator` class is modified to use CuPy arrays, and the `ApplyGainOp` operator is updated to use CuPy functions for array manipulation. The `CudaTxOp` and `CudaRxOp` operators are also updated to work with CuPy arrays.

With CuPy, you can conveniently perform GPU-accelerated computations on multidimensional arrays, making it an excellent choice for high-performance data processing tasks in Holoscan applications.

Please note that CuPy does not fully support certain CUDA APIs, such as `cupy.cuda.driver.occupancyMaxPotentialBlockSize()`. While the driver API may be available ([link](https://github.com/cupy/cupy/pull/2424)), it is currently undocumented ([link](https://docs.cupy.dev/en/stable/reference/cuda.html)) and lacks support for calling the API with RawKernel's pointer ([link](https://github.com/cupy/cupy/issues/2450)), or using CUDA Python's `cuda.cuOccupancyMaxPotentialBlockSize()` Driver API with CuPy-generated RawKernel functions.

Currently, direct assignment of a non-default CUDA stream to HolovizOp in Holoscan applications is not supported without utilizing the `holoscan.resources.CudaStreamPool` resource. CuPy also has limited support for custom stream management, necessitating reliance on the default stream in this context.

For more detailed information, please refer to the following resources:
- [New RawKernel Calling Convention / Kernel Occupancy Functions · Issue #3684 · cupy/cupy · GitHub](https://github.com/cupy/cupy/issues/3684)
- CUDA Stream Support:
  - [Enhancing stream support in CuPy's default memory pool · Issue #8068 · cupy/cupy](https://github.com/cupy/cupy/issues/8068)
  - [cupy.cuda.ExternalStream — CuPy 13.1.0 documentation](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.ExternalStream.html)
