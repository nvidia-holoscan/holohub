# AI Segmentation using MONAI re-trained TotalSegmentator model and CT DICOM images as input

This application uses MONAI re-trained TotalSegmentator model to segment 104 body parts from a DICOM series of a CT scan. It is implemented using Holohub DICOM processing operators and PyTorch inference operators.

## Requirements

- On a [Holohub supported platform](../../README.md#supported-platforms)
- Python 3.8+
- Python packages on [Pypi](https://pypi.org), including but not limited to torch, monai, nibabel, pydicom, highdicom, and others as specified in the requirements file
- Nvidia GPU with at least 14GB memory, for a 200 slice CT series


## Model

This application uses the [MOMAI whole-body segmentation model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation).

## Data

The input for this application is a folder of DICOM image files from a CT series. For testing, CT scan images can be downloaded from [The Cancer Imaging Archive](https://nbia.cancerimagingarchive.net/nbia-search/), subject to [Data Usage Policies and Restrictions](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/)

One such data set, a CT Abdomen series described as `ABD/PANC_3.0_B31f`, is copied and made available [here](https://urm.nvidia.com/artifactory/sw-holoscan-generic/test_data/dicom/TCIA_CT_ABDOMEN/)

## Run Instructions

This application only has Python implementation depending on a set of Python packages from [Pypi](https://pypi.org).

The `PYTHONPATH` environment variable must be set to locate the necessary Holohub medical imaging operators. Furthermore, the AI model as well as input DICOM files must be located via application specific environment variables, namely `HOLOSCAN_MODEL_PATH` and `HOLOSCAN_INPUT_PATH` respectively.

The application output folder can be defined by the environment variable `HOLOSCAN_OUTPUT_PATH`, which defaults to `output`.

There are a few ways to prepare and run the application, and a basic one is described below, with more automated ones coming later.

### Run the application directly in a Holohub Container

[Build and launch the Holohub Container](../../README.md#container-build-recommended), landing in `/workspace/holohub`

Set up the Holohub environment
```bash
./run setup
```

Install Python packages required by the application
```bash
pip install -r applications/imaging_ai_segmentator/requirements.txt
```

Set the `PYTHONPATH` to include the Holohub source folder
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
```

Set the environment variables for the application
```bash
source applications/imaging_ai_segmentator/env_settings.sh
```

Download the AI model from [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation)
```bash
mkdir -p applications/imaging_ai_segmentator/models
pip install gdown
python -m gdown https://drive.google.com/uc?id=1PHpFWboimEXmMSe2vBra6T8SaCMC2SHT -O applications/imaging_ai_segmentator/models/model.pt
```

Download images of a CT series from [TCIA](https://nbia.cancerimagingarchive.net/nbia-search/), or from the Holohub artifactory
```bash
wget --no-parent -r -l 1  "https://urm.nvidia.com/artifactory/sw-holoscan-generic/test_data/dicom/TCIA_CT_ABDOMEN/"
rm -f -r $HOLOSCAN_INPUT_PATH
mkdir -p $HOLOSCAN_INPUT_PATH
cp -r urm.nvidia.com/artifactory/sw-holoscan-generic/test_data/dicom/TCIA_CT_ABDOMEN/ $HOLOSCAN_INPUT_PATH
```

Run the application
```bash
rm -f -r $HOLOSCAN_OUTPUT_PATH
python applications/imaging_ai_segmentator/app.py
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

