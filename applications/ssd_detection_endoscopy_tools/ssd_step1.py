# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools
import os
from argparse import ArgumentParser
from math import sqrt

import cupy as cp
import holoscan as hs
import numpy as np
import torch
import torch.nn.functional as F
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import FormatConverterOp, HolovizOp, InferenceOp, VideoStreamReplayerOp
from holoscan.resources import BlockMemoryPool, MemoryStorageType, UnboundedAllocator

from holohub.aja_source import AJASourceOp

torch.cuda.set_device(torch.device("cuda:0"))

debug_tensor_values_preproc = False


"""
The definitions for calc_iou_tensor(), DefaultBoxes(), dboxes300_coco(), Encoder()
are all from the SSD model training repo:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/utils.py
"""


# This function is from https://github.com/kuangliu/pytorch-ssd.
def calc_iou_tensor(box1, box2):
    """Calculation of IoU based on two boxes tensor,
    Reference to https://github.com/kuangliu/pytorch-src
    input:
        box1 (N, 4)
        box2 (M, 4)
    output:
        IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:, :, :2], be2[:, :, :2])
    # mask1 = (be1[:,:, 0] < be2[:,:, 0]) ^ (be1[:,:, 1] < be2[:,:, 1])
    # mask1 = ~mask1
    rb = torch.min(be1[:, :, 2:], be2[:, :, 2:])
    # mask2 = (be1[:,:, 2] < be2[:,:, 2]) ^ (be1[:,:, 3] < be2[:,:, 3])
    # mask2 = ~mask2

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]
    # *mask1.float()*mask2.float()

    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou


# This function is from https://github.com/kuangliu/pytorch-ssd.
class Encoder(object):
    """
    Inspired by https://github.com/kuangliu/pytorch-src
    Transform between (bboxes, labels) <-> SSD output
    dboxes: default boxes in size 8732 x 4,
        encoder: input ltrb format, output xywh format
        decoder: input xywh format, output ltrb format
    encode:
        input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
        output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
        criteria : IoU threshold of bboexes
    decode:
        input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
        output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
        criteria : IoU threshold of bboexes
        max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def scale_back_batch(self, bboxes_in, scores_in):
        """
        Do scale and transform from xywh to ltrb
        suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = (
            bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        )
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # Transform format to ltrb
        left, top, right, bottom = (
            bboxes_in[:, :, 0] - 0.5 * bboxes_in[:, :, 2],
            bboxes_in[:, :, 1] - 0.5 * bboxes_in[:, :, 3],
            bboxes_in[:, :, 0] + 0.5 * bboxes_in[:, :, 2],
            bboxes_in[:, :, 1] + 0.5 * bboxes_in[:, :, 3],
        )

        bboxes_in[:, :, 0] = left
        bboxes_in[:, :, 1] = top
        bboxes_in[:, :, 2] = right
        bboxes_in[:, :, 3] = bottom

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, criteria=0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, criteria, max_output))
        return output

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200):
        # Reference to https://github.com/amdegroot/ssd.pytorch

        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            # print(score[score>0.90])
            if i == 0:
                continue
            # print(i)

            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0:
                continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []
            # maxdata, maxloc = scores_in.sort()

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        if not bboxes_out:
            return [torch.tensor([]) for _ in range(3)]

        bboxes_out, labels_out, scores_out = (
            torch.cat(bboxes_out, dim=0),
            torch.tensor(labels_out, dtype=torch.long),
            torch.cat(scores_out, dim=0),
        )

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]


class DefaultBoxes(object):
    def __init__(
        self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2
    ):
        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        if order == "xywh":
            return self.dboxes


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


class ProbeOp(Operator):
    def setup(self, spec):
        spec.input("in")
        spec.output("out")
        spec.param("tensor_name", "")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")

        source_video = in_message.get(self.tensor_name)
        if source_video is not None:
            print("Probing " + self.tensor_name + " in " + self.name + " (cupy): ")
            print(cp.asarray(source_video, dtype=cp.float32))
            cp.save("./" + self.name, cp.asarray(source_video, dtype=cp.float32))
            print("Probing " + self.tensor_name + " in " + self.name + " (cupy) shape: ")
            print(cp.shape(source_video))
        op_output.emit(in_message, "out")


class DetectionPostprocessorOp(Operator):
    """Example of an operator post processing the tensor from inference component.
    Following the example of tensor_interop.py and ping.py4

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
    """

    def __init__(self, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")

        locs_tensor = in_message.get("inference_output_tensor_loc")
        labels_tensor = in_message.get("inference_output_tensor_label")

        if locs_tensor is not None:
            locs_pyt = torch.tensor(
                locs_tensor,
            ).cuda()
            # print(f"Received locs_pyt(pytorch): {locs_pyt}")
        if labels_tensor is not None:
            labels_pyt = torch.tensor(
                labels_tensor,
            ).cuda()
            # print(f"Received labels_pyt(pytorch): {labels_pyt}")

        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)
        encoded = encoder.decode_batch(
            locs_pyt, labels_pyt, criteria=0.5, max_output=20
        )  # TIME SINK, TODO: OPTIMIZE

        bboxes, classes, confidences = [
            x.detach().cpu().numpy().astype(np.float32) for x in encoded[0]
        ]

        best = np.argwhere(confidences > 0.3).squeeze()

        # Holoviz takes in rectangles of shapes [1, n/2, 2] where
        # n is the number of rectangles to render. bboxes[idx] has
        # four values in range [0,1], representing [left, top, right, bottom]
        bboxes_output = bboxes[best].reshape(1, -1, 2)

        # for idx in best:
        #    left, top, right, bottom = bboxes[idx] # these four values range from [0, 1]

        # output
        out_message = Entity(context)
        output_tensor = hs.as_tensor(bboxes_output)

        out_message.add(output_tensor, "rectangles")
        op_output.emit(out_message, "out")


class SSDDetectionApp(Application):
    def __init__(self, source="replayer"):
        """Initialize the ssd detection application

        Parameters
        ----------
        source : {"replayer", "aja"}
            When set to "replayer" (the default), pre-recorded sample video data is
            used as the application input. Otherwise, the video stream from an AJA
            capture card is used.
        """

        super().__init__()

        # set name
        self.name = "SSD Detection App"

        # Optional parameters affecting the graph created by compose.
        self.source = source

    def compose(self):
        n_channels = 4  # RGBA
        bpp = 4  # bytes per pixel

        is_aja = self.source.lower() == "aja"
        drop_alpha_block_size = 1920 * 1080 * n_channels * bpp
        drop_alpha_num_blocks = 2

        if is_aja:
            source = AJASourceOp(self, name="aja", **self.kwargs("aja"))
            drop_alpha_block_size = 1920 * 1080 * n_channels * bpp
            drop_alpha_num_blocks = 2
            drop_alpha_channel = FormatConverterOp(
                self,
                name="drop_alpha_channel",
                pool=BlockMemoryPool(
                    self,
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=drop_alpha_block_size,
                    num_blocks=drop_alpha_num_blocks,
                ),
                **self.kwargs("drop_alpha_channel"),
            )
        else:
            source = VideoStreamReplayerOp(self, name="replayer", **self.kwargs("replayer"))

        width_preprocessor = 1920
        height_preprocessor = 1080
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 2
        detection_preprocessor = FormatConverterOp(
            self,
            name="detection_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            **self.kwargs("detection_preprocessor"),
        )

        if debug_tensor_values_preproc is True:
            probe_tensor_before_inf = ProbeOp(
                self, name="probe_tensor_before_inf", tensor_name="source_video"
            )
            probe_tensor_before_preproc = ProbeOp(
                self, name="probe_tensor_before_preproc", tensor_name=""
            )
        detection_inference = InferenceOp(
            self,
            name="detection_inference",
            allocator=UnboundedAllocator(self, name="pool"),
            **self.kwargs("detection_inference"),
        )

        detection_postprocessor = DetectionPostprocessorOp(
            # this is where we write our own post processor in the BYOM process
            self,
            name="detection_postprocessor",
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("detection_postprocessor"),
        )

        detection_visualizer = HolovizOp(
            self,
            name="detection_visualizer",
            tensors=[
                dict(name="", type="color"),
                dict(
                    name="rectangles",
                    type="rectangles",
                    opacity=0.5,
                    line_width=4,
                    color=[1.0, 0.0, 0.0, 1.0],
                ),
            ],
            **self.kwargs("detection_visualizer"),
        )

        if is_aja:
            self.add_flow(source, detection_visualizer, {("video_buffer_output", "receivers")})
            self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
            self.add_flow(drop_alpha_channel, detection_preprocessor)
        else:
            self.add_flow(source, detection_visualizer, {("", "receivers")})

            if debug_tensor_values_preproc is True:
                self.add_flow(source, probe_tensor_before_preproc)
                self.add_flow(probe_tensor_before_preproc, detection_preprocessor)
            else:
                self.add_flow(source, detection_preprocessor)

        if debug_tensor_values_preproc is True:
            self.add_flow(detection_preprocessor, probe_tensor_before_inf)
            self.add_flow(probe_tensor_before_inf, detection_inference, {("out", "receivers")})
        else:
            self.add_flow(detection_preprocessor, detection_inference, {("", "receivers")})
        self.add_flow(detection_inference, detection_postprocessor, {("transmitter", "in")})
        self.add_flow(detection_postprocessor, detection_visualizer, {("out", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="SSD Detection demo application.")
    parser.add_argument(
        "-s",
        "--source",
        choices=["replayer", "aja"],
        default="replayer",
        help=(
            "If 'replayer', replay a prerecorded video. If 'aja' use an AJA "
            "capture card as the source (default: %(default)s)."
        ),
    )
    args = parser.parse_args()

    # point to the config file: "ssd_endo_model.yaml"
    config_file = os.path.join(os.path.dirname(__file__), "ssd_endo_model.yaml")

    app = SSDDetectionApp(source=args.source)
    app.config(config_file)
    app.run()
