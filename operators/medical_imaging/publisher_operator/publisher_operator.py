# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from os import getcwd, makedirs, path
from pathlib import Path
from shutil import copy
from typing import Union

from holoscan.core import Operator


class PublisherOperator(Operator):
    """This Operator publishes the input and segment mask images for the 3rd party Render Server.

    It takes as input the folder path to the input and mask images, in nii, nii.gz, or mhd,
    generates the render config file and the meta data file, then save all in the `publish` folder of the app.
    """

    # The default input folder for saving the generated DICOM instance file.
    DEFAULT_INPUT_FOLDER = Path(getcwd()) / "input"

    # The default output folder for saving the generated DICOM instance file.
    DEFAULT_OUTPUT_FOLDER = Path(getcwd()) / "output"

    def __init__(
        self,
        *args,
        input_folder: Union[str, Path],
        output_folder: Union[str, Path],
        **kwargs,
    ):
        """Class to write DICOM Encapsulated PDF Instance with PDF bytes in memory or in a file.

        Args:
            input_folder (str or Path): The folder to read the input and segment mask files.
            output_folder (str or Path): The folder to save the published files.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        # Need to get the input folder in init until the execution context supports input path.
        self.input_folder = (
            Path(input_folder) if input_folder else PublisherOperator.DEFAULT_INPUT_FOLDER
        )

        # Need to get the output folder in init until the execution context supports output path.
        # Not trying to create the folder to avoid exception on init
        self.output_dir = (
            Path(output_folder) if output_folder else PublisherOperator.DEFAULT_OUTPUT_FOLDER
        )

        super().__init__(*args, **kwargs)

    def compute(self, op_input, op_output, context):
        saved_images_folder = op_input.get("saved_images_folder").path

        if not self.input_folder.is_dir():
            raise ValueError("Expected a folder path for saved_image_folder input")

        density_path, mask_path = self._find_density_and_mask_files(saved_images_folder)
        subs = {"DENSITY_FILE": path.basename(density_path), "MASK_FILE": path.basename(mask_path)}

        publish_folder_path = context.output.get().path.joinpath("publish")
        makedirs(publish_folder_path, exist_ok=True)
        print(f"App publish folder: {publish_folder_path}")

        # Copy over the density and mask files
        copy(density_path, publish_folder_path)
        copy(mask_path, publish_folder_path)

        # Generate the config_render.json from template
        with open(path.join(publish_folder_path, "config_render.json"), "w") as config_render_file:
            config_render_file.write(CONFIG_RENDER_TXT)

        # Replace the file name in the CONFIG_META_TXT and save as config.meta
        config_meta_txt = CONFIG_META_TXT.replace("DENSITY_FILE", subs["DENSITY_FILE"]).replace(
            "MASK_FILE", subs["MASK_FILE"]
        )
        with open(path.join(publish_folder_path, "config.meta"), "w") as config_meta_file:
            config_meta_file.write(config_meta_txt)

    def _find_density_and_mask_files(self, folder_path):
        density_path = None
        mask_path = None

        for matched_file in Path(folder_path).rglob("*"):
            # Need to get the file name in str
            f_name = str(matched_file)
            if path.isfile(f_name):
                if f_name.lower().endswith("_seg.nii.gz"):
                    mask_path = f_name
                    print(f"Mask file path: {mask_path}")
                elif f_name.lower().endswith(".nii.gz"):
                    density_path = f_name
                    print(f"Density file path: {density_path}")
            if density_path and mask_path:
                return density_path, mask_path

        raise ValueError("Cannot find both density and mask nii.gz files.")


CONFIG_META_TXT = """
{
  "data": [
    {
      "file": "DENSITY_FILE",
      "order": "DXYZ"
    },
    {
      "file": "MASK_FILE",
      "order": "MXYZ"
    }
  ],
  "name": "BTCV_highres",
  "settings": "config_render.json"
}
"""

CONFIG_RENDER_TXT = """
{
    "BackgroundLight": {
      "bottomColor": {
        "x": 1,
        "y": 1,
        "z": 1
      },
      "enable": "SWITCH_ENABLE",
      "horizonColor": {
        "x": 1,
        "y": 1,
        "z": 1
      },
      "intensity": 1,
      "show": "SWITCH_DISABLE",
      "topColor": {
        "x": 1,
        "y": 1,
        "z": 1
      }
    },
    "Camera": {
      "eye": {
        "x": 0.25439594946576577,
        "y": 0.28691943829298083,
        "z": 0.7147179382378706
      },
      "fieldOfView": 30,
      "lookAt": {
        "x": -0.03195509341043604,
        "y": 0.006501020775334048,
        "z": -0.016047870948873777
      },
      "name": "Cinematic",
      "pixelAspectRatio": 1,
      "up": {
        "x": -0.11501682740802115,
        "y": 0.9416680547756795,
        "z": -0.31627899713554536
      }
    },
    "CameraAperture": {
      "aperture": 0.0051,
      "autoFocus": "SWITCH_ENABLE",
      "enable": "SWITCH_ENABLE",
      "focusDistance": 9.999999999999999e-06
    },
    "Cameras": [
      {
        "eye": {
          "x": 0.25439594946576577,
          "y": 0.28691943829298083,
          "z": 0.7147179382378706
        },
        "fieldOfView": 30,
        "lookAt": {
          "x": -0.03195509341043604,
          "y": 0.006501020775334048,
          "z": -0.016047870948873777
        },
        "name": "Cinematic",
        "pixelAspectRatio": 1,
        "up": {
          "x": -0.11501682740802115,
          "y": 0.9416680547756795,
          "z": -0.31627899713554536
        }
      },
      {
        "eye": {
          "x": 0,
          "y": 10,
          "z": 0
        },
        "fieldOfView": 30,
        "lookAt": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "name": "SliceTop",
        "pixelAspectRatio": 1,
        "up": {
          "x": 0,
          "y": 0,
          "z": 1
        }
      },
      {
        "eye": {
          "x": 0,
          "y": 0,
          "z": -10
        },
        "fieldOfView": 30,
        "lookAt": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "name": "SliceFront",
        "pixelAspectRatio": 1,
        "up": {
          "x": 0,
          "y": 1,
          "z": 0
        }
      },
      {
        "eye": {
          "x": 10,
          "y": 0,
          "z": 0
        },
        "fieldOfView": 30,
        "lookAt": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "name": "SliceRight",
        "pixelAspectRatio": 1,
        "up": {
          "x": 0,
          "y": 1,
          "z": 0
        }
      },
      {
        "eye": {
          "x": 0,
          "y": 10,
          "z": 0
        },
        "fieldOfView": 30,
        "lookAt": {
          "x": 0,
          "y": 0,
          "z": 0
        },
        "name": "SliceOblique",
        "pixelAspectRatio": 1,
        "up": {
          "x": 0,
          "y": 0,
          "z": 1
        }
      }
    ],
    "DataCrop": {
      "limits": [
        {
          "max": 1,
          "min": 0
        },
        {
          "max": 0.89,
          "min": 0.06
        },
        {
          "max": 1,
          "min": 0
        },
        {
          "max": 1,
          "min": 0.18
        }
      ]
    },
    "DataView": {},
    "Light": [
      {
        "color": {
          "x": 1,
          "y": 1,
          "z": 1
        },
        "direction": {
          "x": -0.3694435400514917,
          "y": -0.8703544378686093,
          "z": -0.3255681544571566
        },
        "enable": "SWITCH_ENABLE",
        "index": 0,
        "intensity": 1.1,
        "position": {
          "x": 0.4,
          "y": 1,
          "z": 0.85
        },
        "size": 0.1
      }
    ],
    "PostProcessDenoise": {
      "depthWeight": 3,
      "enableIterationLimit": "SWITCH_ENABLE",
      "iterationLimit": 500,
      "method": "AI",
      "noiseThreshold": 0.01,
      "radius": 1,
      "spatialWeight": 0.05
    },
    "PostProcessTonemap": {
      "enable": "SWITCH_ENABLE",
      "exposure": 0.5
    },
    "RenderSettings": {
      "interpolationMode": "CATMULLROM",
      "maxIterations": 1000,
      "shadowStepSize": 1,
      "stepSize": 1
    },
    "TransferFunction": {
      "blendingProfile": "MAXIMUM_OPACITY",
      "components": [
        {
          "activeRegions": [
            0
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0.8549019607843137,
            "z": 0.6980392156862745
          },
          "diffuseStart": {
            "x": 1,
            "y": 0.9647058823529412,
            "z": 0.9215686274509803
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 1,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0.06,
          "range": {
            "max": 0.778,
            "min": 0.521
          },
          "roughness": 9.8,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            1
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0.4980392156862745,
            "z": 0
          },
          "diffuseStart": {
            "x": 1,
            "y": 0.4980392156862745,
            "z": 0
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0.01,
          "range": {
            "max": 0.599,
            "min": 0.465
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            2
          ],
          "diffuseEnd": {
            "x": 0.8235294117647058,
            "y": 0.10588235294117647,
            "z": 0.6666666666666666
          },
          "diffuseStart": {
            "x": 0.8235294117647058,
            "y": 0.10588235294117647,
            "z": 0.6666666666666666
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0,
          "range": {
            "max": 0.599,
            "min": 0.471
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            3
          ],
          "diffuseEnd": {
            "x": 0.8235294117647058,
            "y": 0.10588235294117647,
            "z": 0.6666666666666666
          },
          "diffuseStart": {
            "x": 0.8235294117647058,
            "y": 0.10588235294117647,
            "z": 0.6666666666666666
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0,
          "range": {
            "max": 0.579,
            "min": 0.469
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            4
          ],
          "diffuseEnd": {
            "x": 0.8235294117647058,
            "y": 0.5686274509803921,
            "z": 0.09019607843137255
          },
          "diffuseStart": {
            "x": 0.8235294117647058,
            "y": 0.5686274509803921,
            "z": 0.09019607843137255
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0,
          "range": {
            "max": 0.599,
            "min": 0.436
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            5
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0.17647058823529413,
            "z": 0.03137254901960784
          },
          "diffuseStart": {
            "x": 1,
            "y": 0.17647058823529413,
            "z": 0.03137254901960784
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0,
          "range": {
            "max": 0.514,
            "min": 0.38
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            6
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0,
            "z": 0.2901960784313726
          },
          "diffuseStart": {
            "x": 1,
            "y": 0,
            "z": 0.2901960784313726
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SINE",
          "opacityTransition": 0.18,
          "range": {
            "max": 1,
            "min": 0.457
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            7
          ],
          "diffuseEnd": {
            "x": 0.4980392156862745,
            "y": 0,
            "z": 0
          },
          "diffuseStart": {
            "x": 0.4980392156862745,
            "y": 0,
            "z": 0
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.32,
          "opacityProfile": "SINE",
          "opacityTransition": 0.15,
          "range": {
            "max": 0.612,
            "min": 0.123
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            8
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0.8666666666666667,
            "z": 0.49411764705882355
          },
          "diffuseStart": {
            "x": 1,
            "y": 0.8666666666666667,
            "z": 0.49411764705882355
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0,
          "range": {
            "max": 0.494,
            "min": 0.448
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            9
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0,
            "z": 0.40784313725490196
          },
          "diffuseStart": {
            "x": 1,
            "y": 0,
            "z": 0.40784313725490196
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SINE",
          "opacityTransition": 0.16,
          "range": {
            "max": 0.524,
            "min": 0.471
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            10
          ],
          "diffuseEnd": {
            "x": 0.8941176470588236,
            "y": 0.5098039215686274,
            "z": 0.08627450980392157
          },
          "diffuseStart": {
            "x": 0.8941176470588236,
            "y": 0.5098039215686274,
            "z": 0.08627450980392157
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0,
          "range": {
            "max": 0.579,
            "min": 0.479
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            11
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0.07450980392156863,
            "z": 0
          },
          "diffuseStart": {
            "x": 1,
            "y": 0.07450980392156863,
            "z": 0
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0,
          "range": {
            "max": 0.526,
            "min": 0.443
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            12
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0.9333333333333333,
            "z": 0
          },
          "diffuseStart": {
            "x": 1,
            "y": 0.9333333333333333,
            "z": 0
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.51,
          "opacityProfile": "SQUARE",
          "opacityTransition": 0,
          "range": {
            "max": 0.534,
            "min": 0.353
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        },
        {
          "activeRegions": [
            0
          ],
          "diffuseEnd": {
            "x": 1,
            "y": 0,
            "z": 0
          },
          "diffuseStart": {
            "x": 1,
            "y": 0,
            "z": 0
          },
          "emissiveEnd": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStart": {
            "x": 0,
            "y": 0,
            "z": 0
          },
          "emissiveStrength": 0,
          "opacity": 0.5,
          "opacityProfile": "SINE",
          "opacityTransition": 0,
          "range": {
            "max": 0.662,
            "min": 0.479
          },
          "roughness": 90,
          "specularEnd": {
            "x": 1,
            "y": 1,
            "z": 1
          },
          "specularStart": {
            "x": 1,
            "y": 1,
            "z": 1
          }
        }
      ],
      "densityScale": 710,
      "globalOpacity": 1,
      "gradientScale": 110,
      "shadingProfile": "HYBRID"
    },
    "View": {
      "cameraName": "Cinematic",
      "mode": "CINEMATIC"
    }
  }
"""
