# Medical Imaging Operators

Medical image processing and inference operators.

## Overview

This set of operators accelerate the development of medical imaging AI inference application with DICOM imaging network integration by providing the following,

- application classes to automate the inference with MONAI Bundle as well as normal TorchScript models
- classes to load supported AI model from files to detected devices, GPU or CPU
- classes to parse runtime options and well-known environment variables
- DICOM study parsing and selection classes, as well as DICOM instance to volume image conversion
- DICOM instance writers to encapsulate AI inference results in these DICOM OID,
  - DICOM Segmentation
  - DICOM Basic Text Structured Report
  - DICOM Encapsulated PDF
- Surface mesh generation and storage in STL format
- Visualization with [Clara-Viz](https://pypi.org/project/clara-viz/) integration, as needed

## Requirements

This set of operators depends on [Holoscan SDK Python package](https://pypi.org/project/holoscan/), as well as directly on the following,

- [highdicom](https://pypi.org/project/highdicom/)
- [monai](https://pypi.org/project/monai/)
- [nibabel](https://pypi.org/project/nibabel/)
- [numpy](https://pypi.org/project/numpy/)
- [numpy-stl](https://pypi.org/project/numpy-stl/)
- [Pillow](https://pypi.org/project/pillow/)
- [pydicom](https://pypi.org/project/pydicom/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [scikit-image](https://pypi.org/project/scikit-image/)
- [SimpleITK](https://pypi.org/project/SimpleITK/)
- [torch](https://pypi.org/project/torch/)
- [trimesh](https://pypi.org/project/trimesh/)
- [typeguard](https://pypi.org/project/typeguard/)

## Notices

Many of this set of operators are `Derivative Works` of [MONAI Deploy App SDK](https://github.com/Project-MONAI/monai-deploy) under its [Apache-2.0 license](https://github.com/Project-MONAI/monai-deploy/blob/main/LICENSE), and Nvidia employees have been the main contributors to MONAI Deploy App SDK.

The dependency packages' licences can be viewed at their respective links as shown in the above section.
