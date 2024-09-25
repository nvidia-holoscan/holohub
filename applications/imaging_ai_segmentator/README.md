# AI Segmentation using MONAI re-trained TotalSegmentator model and CT DICOM images as input

This application uses MONAI re-trained TotalSegmentator model to segment 104 body parts from a DICOM series of a CT scan. It is implemented using Holohub DICOM processing operators and PyTorch inference operators.

The input is a DICOM CT Series, and the segmentation results are saved as DICOM Segmentation in Part10 storage format, as well as in NIfTI format. The workflow is summarized below,
- load DICOM studies
- select series with application defined rules
- convert DICOM pixel data to 3D volume image
- use MONAI SDK to transform input/output and perform inference
- write results as DICOM Segmentation OID instance, re-using study level metadata from the original DICOM study so that the new instance and series can be associated with the original study

The following is the screenshot of the 3D volume rendering of the segmentation results in NIfTI format.

<img src="resources/segments_3D.png" alt="isolated" width="800"/>

The following is the screenshot of a slice of the segmentation saved in DICOM segmentation instance (without color coding the segments).

<img src="resources/segments_DICOM_slice.png" alt="isolated" width="800"/>

## Requirements

- On a [Holohub supported platform](../../README.md#supported-platforms)
- Python 3.8+
- Python packages on [Pypi](https://pypi.org), including but not limited to torch, monai, nibabel, pydicom, highdicom, and others as specified in the requirements file
- Nvidia GPU with at least 14GB memory, for a 200 slice CT series


## Model

This application uses the [MONAI whole-body segmentation model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation).

## Data

The input for this application is a folder of DICOM image files from a CT series. For testing, CT scan images can be downloaded from [The Cancer Imaging Archive](https://nbia.cancerimagingarchive.net/nbia-search/), subject to [Data Usage Policies and Restrictions](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/)

One such data set, a CT Abdomen series described as `ABD/PANC_3.0_B31f`, was used in testing the application. Other DICOM CT Abdomen series can be downloaded from TCIA as test inputs, and, of course, users' own DICOM seriese shall equally work.

**_Note_**:
Please download, or otherwise make available, DICOM files of a CT Abdomen series and save them in a folder, preferably named `data/imaging_ai_segmentator/dicom` under the project root, as this folder name is used in the examples in the following steps. Manual download scripts are shown in [`Run the Application in Dev Environment`](#run-the-application-in-dev-environment)

### Data Citation

National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). (2018). The Clinical Proteomic Tumor Analysis Consortium Cutaneous Melanoma Collection (CPTAC-CM) (Version 11) [Dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2018.ODU24GZE


## Run Instructions

There are a number of ways to build and run this application, as well as packaging this application as a Holoscan Application Package. The following sections describe each in detail.

### Quick Start Using Holohub Container

This is the simplest and fastest way to see the application in action running as a container. The input DICOM files must first be downloaded and saved in the folder `$PWD/data/imaging_ai_segmentator/dicom`, whereas the PyTorch model is automatically downloaded when container image is built.

Use the following to build and run the application:

```bash
mkdir -p output
rm -rf output/*
./dev_container build_and_run imaging_ai_segmentator --container_args "-v $PWD/output:/var/holoscan/output -v $PWD/data/imaging_ai_segmentator/dicom:/var/holoscan/input"
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

It is strongly recommended a Python virtual environment is used for running the application in dev environment.

This application only has Python implementation depending on a set of Python packages from [Pypi](https://pypi.org), however, a `build_and_install` step is needed to automate organizing Python code and downloading the model.


Set up the Holohub environment, if not already done
```bash
./run setup
```

Set the environment variables for the application
```bash
source applications/imaging_ai_segmentator/env_settings.sh
```

If not already done, download images of a CT series from [TCIA](https://nbia.cancerimagingarchive.net/nbia-search/), unzip if necessary, and save the folder of DICOM files under the folder `$HOLOSCAN_INPUT_PATH`.

Optionally download the AI model from [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation), or wait till the build step to have it downloaded automatically

```bash
mkdir -p $HOLOSCAN_MODEL_PATH
pip install gdown
python -m gdown https://drive.google.com/uc?id=1PHpFWboimEXmMSe2vBra6T8SaCMC2SHT -O $HOLOSCAN_MODEL_PATH/model.pt
```

Install Python packages required by the application
```bash
pip install -r applications/imaging_ai_segmentator/requirements.txt
```

Build and install the application
```bash
./dev_container build_and_install imaging_ai_segmentator
```

Run the application
```bash
rm -f -r $HOLOSCAN_OUTPUT_PATH
python install/imaging_ai_segmentator/app.py
```

**_Note_**
If desired, run the application with explicitly input, output, and/or model folder path, for example
```bash
rm -f -r ./output
python install/imaging_ai_segmentator/app.py -m $HOLOSCAN_MODEL_PATH -i $HOLOSCAN_INPUT_PATH -o ./output
```

Check output
```bash
ls $HOLOSCAN_OUTPUT_PATH
```

There should be a DICOM segmentation file with randomized file name. There should also a `saved_images_folder` containing folder named after the input DICOM series' instance UID, which in turn contains the input and segmentation image files in NIfTI format, e.g.
```bash
applications/imaging_ai_segmentator/output
├── 1.2.826.0.1.3680043.10.511.3.64271669147396658491950188504278234.dcm
└── saved_images_folder
    └── 1.3.6.1.4.1.14519.5.2.1.7085.2626
        ├── 1.3.6.1.4.1.14519.5.2.1.7085.2626.nii
        └── 1.3.6.1.4.1.14519.5.2.1.7085.2626_seg.nii
```

### Run the Application in Dev Container

In this mode, there is no need to `build` and `install`. The Python code will run in its source folders, and both the model and input DICOM files need to be downloaded manually with the scripts provided below.

Also, the `PYTHONPATH` environment variable must be set to locate the necessary Holohub medical imaging operators. The AI model and input DICOM file paths need defined via environment variables, namely `HOLOSCAN_MODEL_PATH` and `HOLOSCAN_INPUT_PATH` respectively, otherwise they must be provided explicitly as command options.


First [Build and launch the Holohub Container](../../README.md#container-build-recommended), landing in `/workspace/holohub`

Set the `PYTHONPATH` to include the Holohub source folder
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
```

Set the environment variables for the application
```bash
source applications/imaging_ai_segmentator/env_settings.sh
```

If not already done, download images of a CT series from [TCIA](https://nbia.cancerimagingarchive.net/nbia-search/),  unzip if necessary, and save the folder of DICOM files under the folder `$HOLOSCAN_INPUT_PATH`.

Optionally download the AI model from [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation), or wait till the build step to have it downloaded automatically

```bash
mkdir -p $HOLOSCAN_MODEL_PATH
pip install gdown
python -m gdown https://drive.google.com/uc?id=1PHpFWboimEXmMSe2vBra6T8SaCMC2SHT -O $HOLOSCAN_MODEL_PATH/model.pt
```

Install Python packages required by the application
```bash
pip install -r applications/imaging_ai_segmentator/requirements.txt
```
Run the application
```bash
rm -f -r $HOLOSCAN_OUTPUT_PATH
python applications/imaging_ai_segmentator/
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
