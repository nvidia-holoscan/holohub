### Volume Loader

The `volume_loader` operator reads 3D volumes from the specified input file.

The operator supports these file formats:
* [MHD (MetaImage)](https://itk.org/Wiki/ITK/MetaIO/Documentation)
  * Detached-header format only (`.mhd` + `.raw`)
* [NIFTI](https://nifti.nimh.nih.gov/)
* [NRRD (Nearly Raw Raster Data)](https://teem.sourceforge.net/nrrd/format.html)
  * [Detached-header format](https://teem.sourceforge.net/nrrd/format.html#detached) only

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
