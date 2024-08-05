# Simple CV-CUDA application

This application demonstrates seamless interoperability between Holoscan tensors and CV-CUDA tensors. The image processing pipeline is just a simple flip of the video orientation.

Note that the C++ version of this application currently requires extra code to handle conversion
back and forth between CV-CUDA and Holoscan tensor types. On the Python side, the conversion is
trivial due to the support for the [DLPack Python
specification](https://dmlc.github.io/dlpack/latest/python_spec.html) in both CV-CUDA and Holoscan.
We provide two [operators](../../operators/cvcuda_holoscan_interop/README.md) to handle the
interoperability between CVCUDA and Holoscan tensors.

# Using the docker file

This application requires a compiled version of [CV-CUDA](https://github.com/CVCUDA/CV-CUDA).
For simplicity a DockerFile is available. To generate the container run:

```bash
./dev_container build --docker_file ./applications/cvcuda_basic/Dockerfile
```

The C++ version of the application can then be built by launching this container and using the provided `run` script.

```bash
./dev_container launch
./run build cvcuda_basic
```

# Running the Application

This application uses the endoscopy dataset as an example. The `run build` command above will automatically download it. This application is then run inside the container.

```bash
./dev_container launch
```

The Python version of the simple CV-CUDA pipeline example can be run via
```
python applications/cvcuda_basic/python/cvcuda_basic.py --data=/workspace/holohub/data/endoscopy
```

or using the run script

```bash
./run launch cvcuda_basic python
```

The C++ version of the simple CV-CUDA pipeline example can then be run via
```
./build/applications/cvcuda_basic/cpp/cvcuda_basic --data=/workspace/holohub/data/endoscopy
```

or using the run script

```bash
./run launch cvcuda_basic cpp
```


## Dev Container

To start the the Dev Container, run the following command from the root directory of Holohub:

```bash
./dev_container vscode cvcuda_basic
```

### VS Code Launch Profiles

#### C++

Use the `**(gdb) cvcuda_basic/cpp**` launch profile configured for this application to debug the application.


#### Python

There are two launch profiles configured for this Python application:

1. **(debugpy) cvcuda_basic/python**: Launch cvcuda_basic using a launch profile that enables debugging of Python code.
2. **(pythoncpp) cvcuda_basic/python**: Launch cvcuda_basic using a launch profile that enables debugging of Python and C++ code.
