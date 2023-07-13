# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
import time

import cupy as cp
import holoscan as hs
from holoscan.conditions import BooleanCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import HolovizOp
from PIL import Image, ImageDraw, ImageFont

OVERSAMPLE_FACTOR = 1
C = 299792458.0
PI = 3.1415926535897932384626433832795
HUGE = 1000000000000.0
FIRST_PULSE_BYTE_OFFSET = 24


# import numpy as cp


def npot(x):
    return 1 << (x - 1).bit_length()


def drop_image(filename, data):
    b = cp.nan_to_num(data)
    b = b - cp.min(b)
    b = b / cp.max(b)
    if cp.__name__ == "cupy":
        b = b.get()
    im = Image.fromarray(cp.uint8(b * 255), "L")
    im = im.save(filename)


class Signal_GeneratorOp(Operator):
    """Read the sample data from file and emit to image generation operators

    Parameters
    ----------
    bfile             : file handle to use for reading inputs
    num_pulses        : maximum number of pulses to emit before entering infinite loop
    num_samples       : number of samples per pulse
    oversample_factor : oversampling factor for optional FFT of input values
    fft_input         : flag indicating whether input data should be fourier transformed before
                        emission
    Target_PRF        : if positive, attempts to set pulse output to the given frequency
    Abort_After       : if positive, the operator will terminate after given pulse count

    Output
    ------
    xyz          : numpy vector representing the location of the receiver for the current pulse
    r0           : Range to center of scene as provided by the input file
    r1           : Range to center of scene as calculated at runtime
    sample_count : number of samples in the pulse
    samples      : numpy vector of length <sample_count> complex values representing return
                   samples

    """

    def __init__(
        self,
        *args,
        bfile,
        num_pulses=-1,
        num_samples=-1,
        oversample_factor=1,
        Input_Filename="",
        Fourier_Transform_Input=-1,
        File_Loop=0,
        Target_PRF=-1,
        Abort_After=-1,
        dtype=cp.int32,
        **kwargs,
    ):
        self.bfile = bfile
        self.num_pulses = num_pulses
        self.num_samples = num_samples
        self.oversample_factor = oversample_factor
        self.fft_input = Fourier_Transform_Input
        self.target_prf = Target_PRF
        self.file_loop = File_Loop
        self.abort_after = Abort_After

        assert num_pulses > 0
        assert num_samples > 0

        print("source num_pulses=", self.num_pulses)
        print("source num_samples=", self.num_samples)
        print("Oversample_factor=", self.oversample_factor)
        print("FFT input = ", self.fft_input)
        print("Target PRF = ", self.target_prf)
        print("file_loop = ", self.file_loop)

        self.loop_count = 0
        self.total_count = 0
        self.dtype = dtype
        super().__init__(*args, **kwargs)
        self.timer_inited = 0
        if Target_PRF > 0:
            self.time_per_pulse = float(1.0) / Target_PRF
        else:
            self.time_per_pulse = 0

    def setup(self, spec: OperatorSpec):
        spec.output("pulse_data")

    def compute(self, op_input, op_output, context):
        current_time = time.perf_counter()
        if self.timer_inited == 0:
            self.first_timer = current_time
            self.timer_inited = 1
        elapsed_time = current_time - self.first_timer
        if self.time_per_pulse > 0:
            if elapsed_time < self.total_count * self.time_per_pulse:
                return
            if elapsed_time > self.time_per_pulse * (self.total_count + 1):
                print(
                    "Late Sending pulse ",
                    self.total_count,
                    " at ",
                    elapsed_time,
                    "scheduled for ",
                    self.time_per_pulse * self.total_count,
                )
        # print ("Sending pulse ", self.count, " at ", elapsed_time)
        xyz_i = cp.fromfile(self.bfile, count=3, dtype=cp.float32)
        xyz = cp.array([xyz_i[0], xyz_i[1], xyz_i[2]], dtype=cp.float64)
        r0 = cp.fromfile(self.bfile, count=1, dtype=cp.float64)
        r1 = cp.linalg.norm(xyz)

        samples = cp.fromfile(self.bfile, count=self.num_samples, dtype=cp.complex64)
        output_samples = self.num_samples
        if self.fft_input == 1:
            if self.oversample_factor > 0:
                Nfft = npot(self.num_samples * self.oversample_factor)
            else:
                Nfft = self.num_samples
            pad_count = Nfft - self.num_samples
            samples = cp.pad(samples, ((0), (pad_count)), "constant", constant_values=((0), (0)))
            samplesa = cp.fft.ifft(samples, norm="backward")
            samples = cp.fft.fftshift(samplesa)
            output_samples = Nfft
        else:
            pass

        out = {"xyz": xyz, "r0": r0, "r1": r1, "sample_count": output_samples, "samples": samples}
        self.loop_count += 1
        self.total_count += 1
        if self.loop_count == self.num_pulses:
            if self.file_loop == 0:
                print("Hanging after source emitter reached end of file @ pulse #", self.num_pulses)
                while 1:
                    pass
            else:
                self.bfile.seek(FIRST_PULSE_BYTE_OFFSET)
                self.loop_count = 0
        if (self.abort_after > 0) & (self.total_count > self.abort_after):
            self.conditions["enabled"].disable_tick()

        op_output.emit(out, "pulse_data")


class BP_Image_FormationOp(Operator):
    """Form an image of the ground plane of interest via back projection.
    this algorithm applies one pulse at a time to all pixels in the image

    Parameters
    ----------
    num_samples : int
        number of samples per pulse
    dr: double
        range difference between adjacent samples
    minf: double
        Minimum frequency in each pulse
        [need to understand this better.  Used for matched filter]

    Input
    ------
    xyz          : numpy vector representing the location of the receiver for the current pulse
    r0           : Range to center of scene as provided by the input file
    r1           : Range to center of scene as calculated at runtime
    sample_count : number of samples in the pulse
    samples      : numpy vector of length <sample_count> complex values representing return samples

    Output
    ------
    xyz          : numpy vector representing the location of the receiver for the most recently
                   consumed pulse
    image        : numpy matrix of complex values representing reflectivity at points of interest

    """

    def __init__(
        self,
        *args,
        num_samples=-1,
        dr=-1,
        Image_Size_X=-1,
        Image_Size_Y=-1,
        Pixel_Spacing=-1,
        Pulses_To_Integrate=-1,
        Algorithm="",
        minf=-1,
        **kwargs,
    ):
        self.count = 0
        assert num_samples > 0
        assert dr > 0
        assert minf > 0
        self.dr_inv = 1 / dr
        self.image_size_x = Image_Size_X
        self.image_size_y = Image_Size_Y
        self.pixel_spacing = Pixel_Spacing
        self.minf = minf
        self.pc_partial = 4.0 * PI * (cp.float64(minf) / C)
        self.num_samples = num_samples
        self.accumulator = cp.zeros([self.image_size_y, self.image_size_x]) + (0 + 0j)
        self.xp = cp.linspace(0, self.num_samples - 1, self.num_samples)
        self.bin_offset = num_samples / 2
        self.pulses_to_integrate = Pulses_To_Integrate
        self.buffer_head = 0
        self.buffer_tail = 0
        self.pulse_buffer = list({} for i in range(Pulses_To_Integrate))
        self.total_time = 0
        min_x = (-self.image_size_x / 2.0 + 0.5) * self.pixel_spacing
        max_y = (self.image_size_y - 1) / 2 * self.pixel_spacing
        xx = cp.linspace(
            min_x, min_x + self.pixel_spacing * (self.image_size_x - 1), self.image_size_x
        )
        yy = cp.linspace(max_y, -max_y, self.image_size_y)
        self.coords = cp.meshgrid(xx, yy)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("signal_in")
        spec.output("signal_out")

    def pulse_bit(self, pulse, factor):
        xyz = pulse["xyz"]
        r0 = pulse["r1"]
        pulse["sample_count"]
        input_samples = pulse["samples"]
        pixel_radius = (
            cp.sqrt((self.coords[0] - xyz[0]) ** 2 + (self.coords[1] - xyz[1]) ** 2 + xyz[2] ** 2)
            - r0
        )
        sample_bin = pixel_radius * self.dr_inv + self.bin_offset
        samples = cp.interp(sample_bin, self.xp, input_samples, left=0.000001, right=0.000001)
        mf = cp.exp(pixel_radius * self.pc_partial * (0 + 1j))
        contribution = mf * samples * factor
        return contribution

    def compute(self, op_input, op_output, context):
        pulse_data = op_input.receive("signal_in")
        xyz = pulse_data["xyz"]
        if (self.count % 100) == 0:
            print("count=", self.count, "head=", self.buffer_head, "total_time=", self.total_time)
            # print (".", end="")
            sys.stdout.flush()
        pulse_data["r1"]
        pulse_data["sample_count"]

        # save incoming pulses in rotating buffer
        this_pulse = self.buffer_head
        self.pulse_buffer[self.buffer_head] = pulse_data
        self.buffer_head += 1
        if self.buffer_head == self.pulses_to_integrate:
            self.buffer_head = 0
        start_time = time.perf_counter()
        new_addition = self.pulse_bit(self.pulse_buffer[this_pulse], 1.0)
        self.accumulator += new_addition
        if self.count > self.pulses_to_integrate:
            self.accumulator -= self.pulse_bit(self.pulse_buffer[self.buffer_head], 1.0)
        stop_time = time.perf_counter()
        self.total_time += stop_time - start_time
        self.count += 1

        out = {"xyz": xyz, "image": self.accumulator}
        op_output.emit(out, "signal_out")


class Image_OutputOp(Operator):
    """Write the complex image as a greymap file representing the magnitude
    of reflectivity at each point.  This method discards phase and is not optimal for
    downstream processing, but makes a good visualization"""

    def __init__(self, *args, Output_Filename_Prefix="", final=-1, **kwargs):
        self.final = final
        self.image_prefix = Output_Filename_Prefix
        print("Setting final=", self.final)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("complex_image")
        spec.output("outputs")
        self.first_xyz = cp.array([-1, -1, -1])
        self.count = 0

    def compute(self, op_input, op_output, context):
        self.count += 1
        input = op_input.receive("complex_image")
        image = input["image"]
        image_size_y = image.shape[0]
        image_size_x = image.shape[1]
        xyz = input["xyz"]
        if ((self.count % 100) > 0) and (self.count > 10):
            return
        if self.first_xyz[0] < 0:
            self.first_xyz = xyz
        b = cp.abs(image)
        v1 = xyz[:2]
        v2 = self.first_xyz[:2]
        cp.arccos(cp.dot(v1, v2) / (cp.linalg.norm(v1) * cp.linalg.norm(v2))) * 180 / PI

        b = cp.log10(cp.abs(b))
        floor = cp.min(b)
        floor = -2
        b = b - floor
        b = b / cp.max(b)
        c = b * 255

        if cp.__name__ == "cupy":
            b = b.get()

        im = Image.fromarray(cp.uint8(b * 255), "L")
        im1 = ImageDraw.Draw(im)
        mf = ImageFont.truetype("font.ttf", 24)
        text = (
            "TX: ("
            + "%4.2f" % xyz[0]
            + ", "
            + "%4.2f" % xyz[1]
            + ") "
            + " Pulses="
            + str(self.count)
        )  # + " angle=" + '%4.2f' % theta
        im1.text((5, 5), text, font=mf)
        # im1.line ((image_size_x/2, image_size_y/2, image_size_x/2+xyz[0],
        #           image_size_y/2-xyz[1]), width=3)
        # im1.line ((image_size_x/2, image_size_y/2, image_size_x/2+self.first_xyz[0],
        #           image_size_y/2-self.first_xyz[1]), width=3)
        e = Image.Image.getdata(im)
        f = cp.asarray(e, dtype=cp.uint8).reshape(image_size_x, image_size_y)
        f = cp.repeat(f[:, :, cp.newaxis], 3, axis=2)
        self.image_prefix + str(self.count) + ".png"
        # im = im.save (filename)
        # print ("File ", filename, " written!")
        print("Image after pulse ", str(self.count), " outputted")
        out_message = Entity(context)
        c = c.astype(dtype=cp.uint8)
        d = cp.repeat(c[:, :, cp.newaxis], 3, axis=2)
        # d=cp.ones([2048,2048,3], dtype=cp.uint8)*255
        if cp.__name__ == "cupy":
            d = d.get()
        out_message.add(hs.as_tensor(f), "pixels")
        plat_x = float(xyz[0])
        plat_y = float(xyz[1])
        print("plat: ", plat_x, plat_y, type(plat_x))
        platform_coords = cp.asarray(
            [
                (0.5, 0.5),
                (plat_x - 0.5, -plat_y + 0.5),
            ],
            dtype=cp.float32,
        )

        # append initial axis of shape 1
        platform_coords = platform_coords[cp.newaxis, :, :]
        if cp.__name__ == "cupy":
            platform_coords = platform_coords.get()
        out_message.add(hs.as_tensor(platform_coords), "platform")

        text = (["label_1", "label_2"],)
        # out_message.add(hs.as_tensor("test"), "textline")
        # out_message.add (0,0,"hello","text")
        op_output.emit(out_message, "outputs")


class SAR_ImagingApp(Application):
    """Synthetic Aperture Radar (SAR) Imaging application.

    Generates one or more SAR images based on input data.

    [Add references to the process]

    The current implementation is somewhat hard coded for the GOTCHA Volumetric SAR datasets.
    A preprocessor is used to convert the provided data sets from a MATLAB-centric format to
    one that is more easily used in real-time, custom application software.

    """

    def compose(self):
        print("SAR_Input:", self.kwargs("SAR_Input"))
        input_filename = str(self.from_config("SAR_Input.Input_Filename"))
        print("input filename=", input_filename)
        fft_input = int(self.from_config("SAR_Input.Fourier_Transform_Input"))
        print("FFT Input=", fft_input)

        self.bfile = open(input_filename, "rb")
        bfile = self.bfile
        np = cp.fromfile(bfile, count=1, dtype=cp.uint32)
        ns = cp.fromfile(bfile, count=1, dtype=cp.uint32)
        self.minf = cp.fromfile(bfile, count=1, dtype=cp.float64)
        self.df = cp.fromfile(bfile, count=1, dtype=cp.float64)

        self.num_pulses = int(np)
        self.num_samples = int(ns)
        if OVERSAMPLE_FACTOR > 0:
            self.Nfft = npot(self.num_samples * OVERSAMPLE_FACTOR)
        else:
            self.Nfft = self.num_samples

        # self.num_pulses = int(self.from_config("SAR_Image_Formation.Pulses_To_Integrate"))
        str(self.from_config("SAR_Image_Formation.Algorithm"))
        source_count = int(self.from_config("SAR_Input.Abort_After"))

        print("file NP=", np)
        print("file NS=", ns)
        print("file minf = ", self.minf)
        print("file df = ", self.df)
        self.dr = C / (self.df * 2) / self.Nfft
        print("dr = ", self.dr)
        print("Range swath = ", self.dr * self.Nfft)
        print("Source Count=", source_count)

        # operators
        signal_generator = Signal_GeneratorOp(
            self,
            BooleanCondition(self, name="enabled"),
            bfile=bfile,
            num_pulses=self.num_pulses,
            num_samples=self.num_samples,
            oversample_factor=OVERSAMPLE_FACTOR,
            name="generator",
            **self.kwargs("SAR_Input"),
        )
        num_samples = self.num_samples
        if (OVERSAMPLE_FACTOR > 0) and (fft_input == 1):
            num_samples = npot(self.num_samples * OVERSAMPLE_FACTOR)
        if 1:
            convolver = BP_Image_FormationOp(
                self,
                name="conv",
                num_samples=num_samples,
                dr=self.dr,
                minf=self.minf,
                **self.kwargs("SAR_Image_Formation"),
            )
        else:
            pass
        printer = Image_OutputOp(
            self, final=self.num_pulses, name="printer", **self.kwargs("SAR_Output")
        )

        visualizer = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        # flows between operators
        self.add_flow(signal_generator, convolver)
        self.add_flow(convolver, printer)
        self.add_flow(printer, visualizer, {("outputs", "receivers")})


if __name__ == "__main__":
    set_log_level(LogLevel.WARN)

    app = SAR_ImagingApp()
    config_file = os.path.join(os.path.dirname(__file__), "app_config.yaml")
    app.config(config_file)
    app.run()
