# Volume Loader

The `volume_loader` operator reads 3D volumes from the specified input file.

## Supported Formats

The operator supports these medical volume file formats:
* [MHD (MetaImage)](https://itk.org/Wiki/ITK/MetaIO/Documentation)
  * Detached-header format only (`.mhd` + `.raw`)
* [NIFTI](https://nifti.nimh.nih.gov/)
* [NRRD (Nearly Raw Raster Data)](https://teem.sourceforge.net/nrrd/format.html)
  * [Attached-header format](https://teem.sourceforge.net/nrrd/format.html) (`.nrrd`)
  * [Detached-header format](https://teem.sourceforge.net/nrrd/format.html#detached) (`.nhdr` + `.raw`)

You must convert your data to one of these formats to load it with `VolumeLoaderOp`. Some third party open source
tools for volume file format conversion include:
- Command Line Tools
  - the [Insight Toolkit (ITK)](https://itk.org/) ([PyPI](https://pypi.org/project/itk/), [Image IO Examples](https://examples.itk.org/src/io/imagebase/))
  - [SimpleITK](https://simpleitk.org/) ([PyPI](https://pypi.org/project/SimpleITK/))
  - [Utah NRRD Utilities (unu)](https://teem.sourceforge.net/unrrdu/)
- GUI Applications
  - [3D Slicer](https://www.slicer.org/)
  - [ImageJ](https://imagej.net/)

## API

#### `holoscan::ops::VolumeLoaderOp`

Operator class to read a volume.

##### Parameters

- **`file_name`**: Volume data file name
  - type: `std::string`
- **`allocator`**: Allocator used to allocate the volume data
  - type: `std::shared_ptr<Allocator>`

##### Outputs

- **`volume`**: Output volume data
  - type: `nvidia::gxf::Tensor`
- **`spacing`**: Physical size of each volume element
  - type: `std::array<float, 3>`
- **`permute_axis`**: Volume axis permutation of data space to world space, e.g. if x and y of a volume is swapped this is {1, 0, 2}
  - type: `std::array<uint32_t, 3>`
- **`flip_axes`**: Volume axis flipping from data space to world space, e.g. if x is flipped this is {true, false, false}
  - type: `std::array<bool, 3>`
- **`extent`**: Physical size of the the volume in world space
  - type: `std::array<float, 3>`
