# Imaging AI Whole Body Segmentation

This application demonstrates the use of the medical imaging operators to build and package an application that parses DICOM images and performs inference using a MONAI model (TotalSegmentator).

![3D Volume Rendering](resources/segments_3D.png)  
_Fig. 1: 3D volume rendering of segmentation results in NIfTI format_

## Overview

This application uses a MONAI re-trained TotalSegmentator model to segment 104 body parts from a DICOM CT series. It is implemented using Holohub DICOM processing operators and PyTorch inference operators.

The input is a DICOM CT series, and the segmentation results are saved as both DICOM Segmentation (Part10 storage format) and NIfTI format. The workflow includes:

- Loading DICOM studies
- Selecting series with application-defined rules
- Converting DICOM pixel data to a 3D volume image
- Using MONAI SDK to transform input/output and perform inference
- Writing results as a DICOM Segmentation OID instance, re-using study-level metadata from the original DICOM study

The segmentation results are saved in both DICOM Segmentation format (Part10 storage) and NIfTI format for visualization and further analysis.

![DICOM Segmentation Slice](resources/segments_DICOM_slice.png)  
_Fig. 2: A slice of the segmentation saved in a DICOM segmentation instance (without color coding the segments)_

## Requirements

- On a [Holohub supported platform](../../README.md#supported-platforms)
- Python 3.8+
- Python packages from [PyPI](https://pypi.org), including:
  - torch
  - monai
  - nibabel
  - pydicom
  - highdicom
  - Other dependencies as specified in [requirements.txt](./requirements.txt)
- NVIDIA GPU with at least 14GB memory (for a 200-slice CT series)

## Data

The input for this application is a folder of DICOM image files from a CT series. For testing, CT scan images can be downloaded from [The Cancer Imaging Archive](https://nbia.cancerimagingarchive.net/nbia-search/), subject to [Data Usage Policies and Restrictions](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).

One such data set, a CT Abdomen series described as `ABD/PANC_3.0_B31f`, was used in testing the application. Other DICOM CT Abdomen series can be downloaded from TCIA as test inputs.

### Data Citation

National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). (2018). The Clinical Proteomic Tumor Analysis Consortium Cutaneous Melanoma Collection (CPTAC-CM) (Version 11) [Dataset]. The Cancer Imaging Archive. <https://doi.org/10.7937/K9/TCIA.2018.ODU24GZE>

## Model

This application uses the [MONAI whole-body segmentation model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation), which can segment 104 body parts from CT scans.

## Build and Run Instructions

### Quick Start Using Holohub Container

This is the simplest and fastest way to run the application:

```bash
./dev_container build_and_run imaging_ai_segmentator
```

**_Note:_** It takes quite a few minutes when this command is run the first time.

By default, the application uses the following directories for input data, model files, and output results:

```bash
HOLOSCAN_INPUT_PATH=<LOCAL_HOLOHUB_PATH>/data/imaging_ai_segmentator/dicom
HOLOSCAN_MODEL_PATH=<LOCAL_HOLOHUB_PATH>/data/imaging_ai_segmentator/models
HOLOSCAN_OUTPUT_PATH=<LOCAL_HOLOHUB_PATH>/build/imaging_ai_segmentator/output
```

Where `<LOCAL_HOLOHUB_PATH>` refers to where you have cloned your Holohub repository and running the `./dev_container` command.

You can modify them by setting the right env variable and mount the right voluems, for instance:

```bash
./dev_container build_and_run imaging_ai_segmentator --container_args "-v /local/output:/my_output -e HOLOSCAN_OUTPUT_PATH=/my_output"
```

The output will be available in the `${HOLOSCAN_OUTPUT_PATH}` directory:

```console
output
├── 1.2.826.0.1.3680043.10.511.3.57591117750107235783166330094310669.dcm
└── saved_images_folder
    └── 1.3.6.1.4.1.14519.5.2.1.7085.2626
        ├── 1.3.6.1.4.1.14519.5.2.1.7085.2626.nii
        └── 1.3.6.1.4.1.14519.5.2.1.7085.2626_seg.nii
```

### Development Environment Setup

You can run the application either in your local development environment or inside the Holohub development container. The steps are nearly identical, with only the first step differing.

1. **Set up the Holohub environment:**

   A. **Within Container:** Build and launch the Holohub Container:

   ```bash
   ./dev_container launch
   ```

   B. **On Bare Metal (not recommended):** Set up the Holohub environment:

   ```bash
   ./run setup
   ```

2. **Set environment variables:**

   ```bash
   source applications/imaging_ai_segmentator/env_settings.sh
   ```

3. **Download test data (if not already done):**
   - Download CT series from [TCIA](https://nbia.cancerimagingarchive.net/nbia-search/)
   - Save DICOM files under `$HOLOSCAN_INPUT_PATH`

4. **Install dependencies:**

   ```bash
   pip install -r applications/imaging_ai_segmentator/requirements.txt
   ```

5. **Build and install the application:**

   ```bash
   ./run build imaging_ai_segmentator
   ```

6. **Run the application:**

   ```bash
   rm -fr $HOLOSCAN_OUTPUT_PATH
   python install/imaging_ai_segmentator/app.py
   ```

   **_Tip:_**
   You can override the default input, output, and model directories by specifying them as command-line arguments. For example:

   ```bash
   python install/imaging_ai_segmentator/app.py -m /path/to/model -i /path/to/input -o /path/to/output
   ```

7. **Check output:**

    ```bash
    ls $HOLOSCAN_OUTPUT_PATH
    ```

## Output

The application generates two types of outputs:

1. DICOM Segmentation file (Part10 storage format)
2. NIfTI format files in the `saved_images_folder`:
   - Original CT scan in NIfTI format
   - Segmentation results in NIfTI format
