# Best Practices to integrate external libraries into Holoscan pipelines
The Holoscan SDK is part of NVIDIA Holoscan, the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

With Holoscan SDK, one can develop an end-to-end GPU accelerated pipeline with RDMA support. However, with increasing requirements on pre-processing/post-processing other than inference only pipeline, the integration with other powerful, GPU-accelerated libraries is needed. 

<div align="center">
<img src="./images/typical_pipeline_holoscan.png" style="border: 2px solid black;">
</div>

The datatype in Holoscan SDK is defined as [Tensor](https://docs.nvidia.com/holoscan/sdk-user-guide/generated/classholoscan_1_1tensor.html) which is a multi-dimensional array of elements of a single data type. The Tensor class is a wrapper around the [DLManagedTensorCtx](https://docs.nvidia.com/holoscan/sdk-user-guide/generated/structholoscan_1_1dlmanagedtensorctx.html#structholoscan_1_1DLManagedTensorCtx) struct that holds the DLManagedTensor object. It also supports both DLPack and NumPy’s array interface (__array_interface__ and __cuda_array_interface__) so that it can be used with other Python libraries such as [CuPy](https://docs.cupy.dev/en/stable/user_guide/interoperability.html), [PyTorch](https://github.com/pytorch/pytorch/issues/15601), [JAX](https://github.com/google/jax/issues/1100#issuecomment-580773098), [TensorFlow](https://github.com/tensorflow/community/pull/180), and [Numba](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html).

In this tutorial, we will show how to integrate the libraries below into Holoscan applications in Python:
- [Integrate RAPIDS **cuCIM** library](#integrate-rapids-cucim-library)
- [Integrate RAPIDS **CV-CUDA** library](#integrate-cv-cuda-library)
- [Integrate **OpenCV with CUDA Module**](#integrate-opencv-with-cuda-module)


## Integrate RAPIDS cuCIM library
[RAPIDS cuCIM](https://github.com/rapidsai/cucim) (Compute Unified Device Architecture Clara IMage) is an open-source, accelerated computer vision and image processing software library for multidimensional images used in biomedical, geospatial, material and life science, and remote sensing use cases.

See the supported Operators in [cuCIM documentation](https://docs.rapids.ai/api/cucim/stable/).

cuCIM offers interoperability with CuPy. We can initialize CuPy arrays directly from Holoscan Tensors and use the arrays in cuCIM operators for processing without memory transfer between host and device. 

### Installation
Follow the [cuCIM documentation](https://github.com/rapidsai/cucim?tab=readme-ov-file#install-cucim) to install the RAPIDS cuCIM library.

### Sample code 
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

## Integrate CV-CUDA library
[CV-CUDA](https://github.com/CVCUDA/CV-CUDA) is an open-source, graphics processing unit (GPU)-accelerated library for cloud-scale image processing and computer vision developed jointly by NVIDIA and the ByteDance Applied Machine Learning teams. CV-CUDA helps developers build highly efficient pre- and post-processing pipelines that can improve throughput by more than 10x while lowering cloud computing costs.

See the supported CV-CUDA Operators in the [CV-CUDA developer guide](https://github.com/CVCUDA/CV-CUDA/blob/main/DEVELOPER_GUIDE.md)

### Installation
Follow the [CV-CUDA documentation](https://cvcuda.github.io/installation.html) to install the CV-CUDA library.

Requirement: CV-CUDA >= 0.2.1 (From which version DLPack interop is supported)

### Sample code 
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

## Integrate OpenCV with CUDA Module
[OpenCV](https://opencv.org/) (Open Source Computer Vision Library) is a comprehensive open-source library that contains over 2500 algorithms covering Image & Video Manipulation, Object and Face Detection, OpenCV Deep Learning Module and much more. 

[OpenCV also supports GPU acceleration](https://docs.opencv.org/4.8.0/d2/dbc/cuda_intro.html), includes a CUDA module which is a set of classes and functions to utilize CUDA computational capabilities. It is implemented using NVIDIA CUDA Runtime API and provides utility functions, low-level vision primitives, and high-level algorithms.

### Installation
Prerequisites:
- OpenCV >= 4.8.0 (From which version, OpenCV GpuMat supports initialization with GPU Memory pointer)

Install OpenCV with its CUDA module following the guide in [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib/tree/4.x) 

We also recommend referring to the [Holoscan Endoscopy Depth Estimation application container](https://github.com/nvidia-holoscan/holohub/blob/main/applications/endoscopy_depth_estimation/Dockerfile) as an example of how to build an image with Holoscan SDK and OpenCV CUDA.  

### Sample code
The datatype of OpenCV is GpuMat which implements neither the __cuda_array_interface__ nor the standard DLPack. To achieve the end-to-end GPU accelerated pipeline / application, we need to implement 2 functions to convert the GpuMat to CuPy array which can be accessed directly with Holoscan Tensor and vice versa. 

Refer to the [Holoscan Endoscopy Depth Estimation sample application](https://github.com/nvidia-holoscan/holohub/tree/main/applications/cvcuda_basic) for an example of how to use the OpenCV operator with Holoscan SDK.

1. Conversion from GpuMat to CuPy Array

The GpuMat object of OpenCV Python bindings provides a cudaPtr method which can be used to access the GPU memory address of a GpuMat object. This memory pointer can be utilized to initialize a CuPy array directly, allowing for efficient data handling by avoiding unnecessary data transfers between the host and device. 

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

Note: In this function we used the [UnownedMemory](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.UnownedMemory.html#cupy.cuda.UnownedMemory) API to create the CuPy array. In some cases, the OpenCV operators will allocate new device memory which needs to be handled by CuPy and the lifetime is not limited to one operator but the whole pipeline. In this case, the CuPy array initiated from the GpuMat shall know the owner and keep the reference to the object. Check the CuPy documentation for more detail on [CuPy interoperability](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#device-memory-pointers).

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

