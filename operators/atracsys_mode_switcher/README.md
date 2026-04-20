# Atracsys Mode Switcher Operator

`AtracsysModeSwitcherOp` is the reusable visualization operator for the Atracsys visualizer app.
It consumes replayed or live visible, infrared, structured-light, and marker-pose streams, then
emits the currently selected base image, projected overlays, point clouds, and Holoviz text specs.

The operator keeps the Atracsys interaction model inside one reusable package:

- keyboard-driven mode changes for interactive runs
- placeholder outputs when a stream is not active yet
- marker geometry projection onto visible or infrared frames
- structured-light point rendering and tracking overlays
- hardware-mode commands for the optional live camera source

Inputs:

- `in_visible_base`
- `in_ir_base`
- `in_structured_points`
- `in_marker_poses`

Outputs:

- `out_base`
- `out_overlay`
- `out_marker_points`
- `out_points`
- `out_mode_text`
- `out_fiducial_text_coords`
- `out_specs`
- `out_hw_cmd`

The application is responsible for providing a valid camera calibration and the marker geometry file.
