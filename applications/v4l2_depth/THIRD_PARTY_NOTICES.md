# Third-party notices

The `v4l2_depth` live and loopback CMake builds prepare a Depth Anything V2 Small model artifact
under `data/v4l2_depth`; the container image does not include it.

- Depth Anything V2 source and the Small checkpoint are distributed under the Apache License 2.0.
  The pinned source revision and checkpoint provenance are recorded in the model-preparation
  script.
- The ONNX export helper is derived from `spacewalk01/depth-anything-tensorrt` and is distributed
  under the MIT License.
- The documentation screenshot is derived from the Pexels video
  [A Woman Running on a Pathway](https://www.pexels.com/video/a-woman-running-on-a-pathway-5823544/)
  and is used under the [Pexels license](https://www.pexels.com/license/).

Copies of the Apache-2.0 and MIT license texts are written to `data/v4l2_depth/licenses`. The Pexels
license remains available at the link above.
