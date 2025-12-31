# BCI Visualization

## Running
1. Download data
   * Please download data from [here](https://drive.google.com/drive/folders/1RpQ6UzjIZAr90FdW9VIbtTFYR6-up7w2) and put everything under `data/bci_visualization`.
   * Includes an activation volume and a segmentation volume.
2. Run application
   ```bash
   ./holohub run bci_visualization
   ```
   This command will build the docker and run the application.

## Expected Results

![Example output for BCI Visualization](docs/brain.png)
### Components

1. **VoxelStreamToVolumeOp** (`operators/voxel_stream_to_volume/voxel_stream_to_volume.py`): Bridge operator that converts sparse voxel data to dense volume
   - Inputs: `affine_4x4`, `hb_voxel_data`
   - Outputs: `volume`, `spacing`, `permute_axis`, `flip_axes`, `mask_volume`, `mask_spacing`, `mask_permute_axis`, `mask_flip_axes`

2. **VolumeRendererOp**: ClaraViz-based volume renderer
   - Renders the 3D volume with transfer functions
   - Supports interactive camera control

3. **HolovizOp**: Interactive visualization
   - Displays the rendered volume
   - Provides camera pose feedback


## Data Flow

```
Reconstruction → VoxelStreamToVolume → VolumeRenderer → Holoviz
                                             ↑               ↓
                                             └─── camera ────┘
```

## Volume Renderer Configuration
Here are some of the important config:
1. `timeSlot`: Rendering time in ms. The longer the better the quality.
2. `TransferFunction`

   a. `activeRegions`:  (0: SKIN, 1: SKULL, 2: CSF, 3: GRAY MATER, 4: WHITE MATTER, 5: AIR). Here, we select [3, 4] for our ROI. Set everything else as opacity=0 (default).
   
   b. `blendingProfile`: If there're overlapping components configured, how to blend.

   c. `range`: Volume's value within this range to be configured. 

   d. `opacityProfile`: How opacity is being applied within this range. Select `Square` for constant.

   e. `diffuseStart/End`: The component's color. Linear interpolation between start/end.

   Note: there are three components configured. First is for overall ROI, set color as white. Second is for positive activation, set as red. Third is for negative activation, set as blue.