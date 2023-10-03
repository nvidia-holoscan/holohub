# Simple CV-CUDA application

This application demonstrates seamless interoperability between Holoscan tensors and CV-CUDA tensors. The image processing pipeline is just a simple flip of the video orientation.

# Using the docker file
This application requires a compiled version of [CV-CUDA](https://github.com/CVCUDA/CV-CUDA).
For simplicity a DockerFile is available. To generate the container run:

```bash
./dev_container build --docker_file ./applications/cvcuda_basic/Dockerfile
```

# Running the Application

This application is then run inside the container:

```bash
./dev_container launch
```

This application uses the endoscopy dataset as an example. To automatically download the dataset, please run:

```bash
./run build cvcuda_basic
```

The simple CV-CUDA pipeline example can then be run via
```
python applications/cvcuda_basic/python/cvcuda_basic.py --data=/workspace/holohub/data/endoscopy
```

or using the run script

```bash
./run launch cvcuda_basic python
```
