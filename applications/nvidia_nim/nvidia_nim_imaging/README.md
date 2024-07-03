# NVIDIA NIM Imaging with Vista-3D

This is a sample application demonstrates the use of Vista-3D NVIDIA Inference Microservice (NIM).

The application instructs the Vista-3D NIM API to process the given dataset and then downloads and extracts the results of a segmentation NRRD file onto a local directory.

### Quick Start

1. Add API key in `nvidia_nim.yaml`
2. `./dev_container build_and_run nvidia_nim_chat`

## Configuring the sample application

Use the `nvidia_nim.yaml` configuration file to configure the sample application:

### Connection Information

```
nim:
  base_url: https://integrate.api.nvidia.com/v1
  api_key:

```

`base_url`: The URL of your NIM instance. Defaults to NVIDIA hosted NIMs.
`api_key`: Your API key to access NVIDIA hosted NIMs.


## Build and Run the sample application

```
# Build the Docker images from the root directory of Holohub
./dev_container build_and_run nvidia_nim_imaging
```

## Display the Results

In this section we will show how to view the sample data and segmentation results returned from Vista-3D.

1. Decompress the sample data volume:
   ```
   gzip -d build/nvidia_nim_imaging/applications/nvidia_nim/nvidia_nim_imaging/sample.nii.gz
   ```
2. Download 3D Slicer: https://download.slicer.org/
3. Decompress and launch 3D Slicer
   ```
   tar -xvzf Slicer-5.6.2-linux-amd64.tar.gz
   ```
4. Locate the sample data volume and the segmentation results in `build/nvidia_nim_imaging/applications/nvidia_nim/nvidia_nim_imaging`
   ```
   drwxr-xr-x 3 vicchang domain-users     4096 Jul  3 11:41 ./
   drwxr-xr-x 4 vicchang domain-users     4096 Jul  3 11:40 ../
   -rw-r--r-- 1 vicchang domain-users 27263336 Jul  3 11:41 1af55a20-e114-4b85-8098-d5c35f00e351.response
   -rw-r--r-- 1 vicchang domain-users 33037057 Jul  3 11:40 sample.nii
   ```
   Rename the `******.response` file to `sample.nrrd`, e.g.:
   ```
   mv build/nvidia_nim_imaging/applications/nvidia_nim/nvidia_nim_imaging/1af55a20-e114-4b85-8098-d5c35f00e351.response build/nvidia_nim_imaging/applications/nvidia_nim/nvidia_nim_imaging/sample.nrrd
   ```
5. In 3D Slicer, click *File*, *Add Data* and click *Choose File(s) to Add*.
   From the *Add Data into the scene* dialog, find and add the `sample.nii` file and the `sample.nrrd` file.
   For the `sample.nrrd` file, select *Segmentation* and click *Ok*.
   ![](./assets/3d-slicer-1.png)
6. 3D Slicer shall display the volume and the segmentation results as shown below:
   ![](./assets/3d-slicer-2.png)