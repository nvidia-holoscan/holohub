# Hugging face pipeline using VISTA3D model and CT images as input

This application illustrate how to integrate a hugging face pipeline into a holoscan application to perform the segment everything of a CT scan. The underlying segmentation model is a pre-trained MONAI model called VISTA3D.

The input is a NIfTI CT image and the segmentation results are saved as the same NIfTI format. The workflow is summarized below,
- load NIfTI images
- use the hugging face pipeline operator to perform the segmentation
- write results as NIfTI to given path

The following is the screenshot of the 3D volume rendering of the segmentation results in NIfTI format.

<img src="resources/segments_3D.png" alt="isolated" width="800"/>

The following is the screenshot of a slice of the segmentation saved in DICOM segmentation instance (without color coding the segments).

<img src="resources/segments_DICOM_slice.png" alt="isolated" width="800"/>

## Requirements

- On a [Holohub supported platform](../../README.md#supported-platforms)
- Python 3.8+
- Python packages on [Pypi](https://pypi.org), including but not limited to torch, monai, nibabel, pydicom, highdicom, and others as specified in the requirements file
- Nvidia GPU with at least 24GB memory, for a 200 slice CT series


## Model

This application uses the [MONAI VISTA3D model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/vista3d).

## Data

The input for this application is a NIfTI CT image. For testing, CT scan images can be downloaded from [the Decathelon spleen dataset](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar).

One such data set, a CT image described as `spleen_1.nii.gz`, was used in testing the application. Other NIfTI images can be selected from the dataset too.

**_Note_**:
Please download, or otherwise make available, NIfTI files of a CT images and save them in a folder, preferably named `data/hugging_face_pipeline/image` under the project root.
### Data Citation

Amber L. Simpson, Michela Antonelli, Spyridon Bakas, Michel Bilello, Keyvan Farahani, Bram van Ginneken, Annette Kopp-Schneider, Bennett A. Landman, Geert Litjens, Bjoern Menze, Olaf Ronneberger, Ronald M. Summers, Patrick Bilic, Patrick F. Christ, Richard K. G. Do, Marc Gollub, Jennifer Golia-Pernicka, Stephan H. Heckers, William R. Jarnagin, Maureen K. McHugo, Sandy Napel, Eugene Vorontsov, Lena Maier-Hein, & M. Jorge Cardoso. (2019). A large annotated medical image dataset for the development and evaluation of segmentation algorithms.


## Run Instructions

There are a number of ways to build and run this application, as well as packaging this application as a Holoscan Application Package. The following sections describe each in detail.

### Quick Start Using Holohub Container

This is the simplest and fastest way to see the application in action running as a container. The input DICOM files must first be downloaded and saved in the folder `$PWD/data/hugging_face_pipeline/image`, whereas the PyTorch model is automatically downloaded when container image is built.

Use the following to build and run the application:

```bash
mkdir -p output
rm -rf output/*
./dev_container build_and_run hugging_face_pipeline --container_args "-v $PWD/output:/var/holoscan/output -v $PWD/data/hugging_face_pipeline/image:/var/holoscan/input"
```

Once the command completes, please check the output folder for the results, e.g.
```
output
├── 1.2.826.0.1.3680043.10.511.3.57591117750107235783166330094310669.dcm
└── saved_images_folder
    └── 1.3.6.1.4.1.14519.5.2.1.7085.2626
        ├── 1.3.6.1.4.1.14519.5.2.1.7085.2626.nii
        └── 1.3.6.1.4.1.14519.5.2.1.7085.2626_seg.nii

2 directories, 3 files
```

**_Note_**
It takes quite a few minutes when this command is run the first time.

### Run the Application in Dev Environment


### Run the Application in Dev Container

In this mode, there is no need to `build` and `install`. The Python code will run in its source folders, and the input files need to be downloaded manually to container path "$HOME/data/dataset".

Also, the `PYTHONPATH` environment variable must be set to locate the necessary Holohub medical imaging operators. The input NIfTI file path needs to be defined via environment variable `HOLOSCAN_INPUT_PATH`.

First [Build and launch the Holohub Container](../../README.md#container-build-recommended), landing in `/workspace/holohub`

Set the `PYTHONPATH` to include the Holohub source folder
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
```

Set the environment variables for the application
```bash
source applications/hugging_face_pipeline/env_settings.sh
```

If not already done, download NIfTI images from [Spleen_Task09](https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar) and save the folder to `$DATASET_PATH`.
```
cd $DATASET_PATH
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
```

Install Python packages required by the application
```bash
pip install -r applications/hugging_face_pipeline/requirements.txt
```

Run the application
```bash
rm -f -r $HOLOSCAN_OUTPUT_PATH
python applications/hugging_face_pipeline/
```

Check output
```bash
ls $HOLOSCAN_OUTPUT_PATH
```

## Packaging the Application for Distribution

With Holoscan CLI, an applications built with Holoscan SDK can be packaged into a Holoscan Application Package (HAP), which is essentially a Open Container Initiative compliant container image. An HAP is well suited to be distributed for deployment on hosting platforms, be a Docker Compose, Kubernetes, or else. Please refer to [Packaging Holoscan Applications](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_packager.html) in the User Guide for more information.

This example application provides all the necessary contents for HAP packaging, and the specific commands are revealed by the specific commands.

**_Note_**

The prerequisite is that the application `build_and_install` has been performed to stage the source and AI model files for packaging.

```
source applications/imaging_ai_segmentator/packageHAP.sh
```
