"""
SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.metrics import CumulativeIterationMetric, DiceMetric
from monai.metrics.utils import do_metric_reduction, ignore_background, is_binary_tensor
from monai.networks import eval_mode
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    EnsureTyped,
    LoadImaged,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from monai.utils import MetricReduction


def get_data(folder):
    """
    Get data name in given folder and return them in dict format for dataset generating.

    Args:
        folder: a folder that contains images and their labels.

    Returns:
        a list of dicts with key words 'image' and 'label'.
    """
    images = sorted(glob(os.path.join(folder, "*[!seg].jpg")))
    labels = [x.replace(".jpg", "_seg.jpg") for x in images]
    for label in labels:
        assert os.path.exists(label)
    return [{"image": i, "label": L} for i, L in zip(images, labels)]


def get_data_json(fn, data_path):
    """
    Get data name in given folder and return them in dict format for dataset generating.

    Args:
        folder: a folder that contains images and their labels.

    Returns:
        a list of dicts with key words 'image' and 'label'.
    """
    # fn = "all_data.json"
    with open(data_path + fn, "r") as f:
        data_info = json.load(f)

    update_paths(data_info, data_path, fn)

    return data_info


def update_paths(info, data_path, fn):
    if "val.json" == fn:
        for entry in info:
            entry["image"] = entry["image"].replace(
                "/home/liubin/data/tooltrack_data/dataset_220809_merge/clean_up_iter_3/samples/",
                data_path,
            )
            entry["label"] = entry["label"].replace(
                "/home/liubin/data/tooltrack_data/dataset_220809_merge/clean_up_iter_3/samples/",
                data_path,
            )
            entry["root"] = entry["root"].replace(
                "/home/liubin/data/tooltrack_data/dataset_220809_merge/clean_up_iter_3/samples/",
                data_path,
            )
    else:
        for entry in info:
            entry["image"] = entry["image"].replace(
                "/home/liubin/data/tooltrack_data/dataset_220809_merge/baseline_label/trainset/",
                data_path,
            )
            entry["label"] = entry["label"].replace(
                "/home/liubin/data/tooltrack_data/dataset_220809_merge/baseline_label/trainset/",
                data_path,
            )
            entry["root"] = entry["root"].replace(
                "/home/liubin/data/tooltrack_data/dataset_220809_merge/baseline_label/trainset",
                data_path,
            )


def get_trans(phase="train"):
    """
    Get pre-training transforms by given phase.

    Args:
        phase: dataset phase, e.g. 'train', 'valid', 'test', default to train.

    Returns:
        transforms for according dataset
    """
    keys = ["image", "label"]
    if "train" == phase:
        transforms = Compose(
            [
                LoadImaged(keys),
                AsChannelFirstd("image"),
                AddChanneld("label"),
                Resized(keys, spatial_size=(736, 480)),
                ScaleIntensityd(keys),
                RandRotated(keys, range_x=np.pi, prob=0.8, mode=["bilinear", "nearest"]),
                RandZoomd(
                    keys,
                    min_zoom=0.8,
                    max_zoom=1.2,
                    prob=0.2,
                    mode=["bilinear", "nearest"],
                ),
                ToTensord(keys),
            ]
        )
    else:
        transforms = Compose(
            [
                LoadImaged(keys),
                AsChannelFirstd("image"),
                AddChanneld("label"),
                Resized(
                    keys=["image", "label"],
                    spatial_size=[736, 480],
                    mode=["bilinear", "nearest"],
                ),
                ScaleIntensityd(keys),
                EnsureTyped(keys),
            ]
        )
    return transforms


def get_ds_and_dl_nocache(data, transforms, shuffle=False):
    """
    Get nocache version dataset and dataloader for a test set.

    Args:
        data: a data dict that contains path of images and labels.
        transforms: pre-train transforms for images and labels.
        shuffle: is shuffle the data, default to false.
    Returns:
        dataset and dataloader.
    """
    ds = Dataset(data=data, transform=transforms)
    dl = DataLoader(ds, batch_size=1, num_workers=1, shuffle=shuffle)
    return ds, dl


def get_ds_and_dl(data, transforms, shuffle=False):
    """
    Get cache version dataset and dataloader for a test set.

    Args:
        data: a data dict that contains path of images and labels.
        transforms: pre-train transforms for images and labels.
        shuffle: is shuffle the data, default to false.

    Returns:
        dataset and dataloader.
    """
    ds = CacheDataset(data=data, transform=transforms, cache_rate=0.5, num_workers=4)
    # dl = DataLoader(ds, batch_size=8, num_workers=4, shuffle=shuffle)
    dl = DataLoader(
        ds, batch_size=8, num_workers=8, shuffle=shuffle
    )  # collate_fn=pad_list_data_collate
    return ds, dl


def get_ds_and_dl_by_data_path(data_path, phase, transforms, train_perc=100):
    """
    Get dataset and dataloader for all sets.

    Args:
        data_path: root path for all images.
        transforms: pre-train transforms for images and labels.
        shuffle: is shuffle the data.

    Returns:
        dataset and dataloader.
    """
    data_dir = os.path.join(data_path, phase)
    file_dict = get_data(data_dir)
    print("$$$", len(file_dict))
    if (train_perc == 100) & ("train" == phase):
        return get_ds_and_dl(file_dict, transforms, True)
    if (train_perc != 100) & ("train" == phase):
        print(f"### NOTE: Training with {train_perc}% of  the data ###")
        random.seed(42)
        file_dict = random.sample(file_dict, int(train_perc / 100 * len(file_dict)))
        return get_ds_and_dl(file_dict, transforms, True)
    elif "valid" == phase:
        return get_ds_and_dl(file_dict, transforms, False)
    else:
        return get_ds_and_dl_nocache(file_dict, transforms)


def get_ds_and_dl_by_json(data_path, phase, transforms, train_perc=100):
    """
    Get dataset and dataloader for all sets.

    Args:
        data_path: root path for all images.
        transforms: pre-train transforms for images and labels.
        shuffle: is shuffle the data.

    Returns:
        dataset and dataloader.
    """
    if phase == "train":
        fn = "train.json"
    elif phase == "valid":
        fn = "val.json"
    elif phase == "test":
        fn = "test.json"

    file_dict = get_data_json(fn, data_path)

    # data_dir = os.path.join(data_path, phase)
    # file_dict = get_data(data_dir)
    print("$$$ Full Dataset size", len(file_dict))
    if (train_perc == 100) & ("train" == phase):
        return get_ds_and_dl(file_dict, transforms, True)
    if (train_perc != 100) & ("train" == phase):
        print(f"### NOTE: Training with {train_perc}% of  the data ###")
        random.seed(42)
        file_dict = random.sample(file_dict, int(train_perc / 100 * len(file_dict)))
        return get_ds_and_dl(file_dict, transforms, True)
    elif "valid" == phase:
        return get_ds_and_dl(file_dict, transforms, False)
    else:
        return get_ds_and_dl_nocache(file_dict, transforms)


def get_dlds_dict(data_path, phase_list, train_perc=100):
    """
    Generating train, valid and test datasets&dataloader
    and putting them into a dict.

    Args:
        datapath: root path of images.
        phase_list: a set name list.

    Returns:
        two dicts contains train, valid and test datasets and dataloaders.
    """
    dataset_dict = dict()
    dataloader_dict = dict()
    for phase in phase_list:
        data_trans = get_trans(phase)
        # dataset_dict[phase], dataloader_dict[phase] = get_ds_and_dl_by_data_path(
        #     data_path, phase, data_trans, train_perc
        # )
        dataset_dict[phase], dataloader_dict[phase] = get_ds_and_dl_by_json(
            data_path, phase, data_trans, train_perc
        )
    return dataset_dict, dataloader_dict


def imsave(ims, phase, save_path):
    """
    Saving imgs in dict given by check_set_and_save_function

    Args:
        ims: an image dict contains images and labels.
        phase: the dataset phase of these images.
        save_path: a path to save these images.
    """
    nrow = len(ims)
    ncol = len(ims[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3), facecolor="white")
    for i, im_dict in enumerate(ims):
        for j, (title, im) in enumerate(im_dict.items()):
            if isinstance(im, torch.Tensor):
                im = im.detach().cpu().numpy()
            # If RGB put channel to end. Else, average across channel dim
            im = np.moveaxis(im, 0, -1) if im.shape[0] == 3 else np.mean(im, axis=0)

            ax = axes[j] if len(ims) == 1 else axes[i, j]
            ax.set_title(f"{phase + ' ' + title}\n{im.shape}")
            im_show = ax.imshow(im)
            ax.axis("off")
            fig.colorbar(im_show, ax=ax)
    save_name = os.path.join(save_path, phase + "_check.png")
    plt.savefig(save_name, format="png")


def check_set_and_save(phase, ds, save_path, size=5, replace=False):
    """
    Checking the prepared dataset and save the images with labels
    by random choice.

    Args:
        phase: the dataset phase for saving.
        ds: the dataset for reading image.
        save_path: the path for saving random images.
        size: number of images need to be saved.
        replace: whether random choice replace the dataset image.
    """
    to_imshow = []
    for data in np.random.choice(ds, size=size, replace=replace):
        to_imshow.append({"image": data["image"], "label": data["label"]})
    imsave(to_imshow, phase, save_path)


def get_metrics():
    """
    Getting the metric objects for evaluating the model.

    Returns:
        metrics for evaluating the model.
    """
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = IoUMetric(include_background=False, reduction="mean")
    return dice_metric, iou_metric


def get_post_trans():
    """
    Getting the post transforms needed to perform on the labels and \
    images after doing inference.

    Returns:
        transforms for post processing.
    """
    post_trans = Compose(
        [
            AsDiscrete(to_onehot=2, argmax=True),
        ]
    )

    post_label = Compose(
        [
            AsDiscrete(to_onehot=2),
        ]
    )
    return post_trans, post_label


def save_training_curve(curve_data, save_path):
    """
    Saving the loss curve and metrics curves produced during training.

    Args:
        cureve_data: a dict that recorded relevant values during training.
        save_path: a path to save the curve image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="white")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")

    for i in ["train_loss", "val_dice", "val_iou"]:
        ax.plot(curve_data["epoch"], curve_data[i], label=i)
    save_name = os.path.join(save_path, "training_cur.png")
    plt.savefig(save_name, format="png")


def infer_seg(images, model, post_trans):
    """
    Do inference of images on a trained model and post transform on result images.

    Args:
        images: images need to do inference.
        model: a trained model put images in.
        post_trans: post transform for inference results.

    Returns:
        results list after inference and posttransform.
    """
    return [post_trans(i) for i in decollate_batch(model(images))]


def get_test_metrics(dataloader, phase, model, device):
    """
    Do test on given dataset.

    Args:
        dataloader: a dataloader to be test on.
        phase: the phase of the input dataset.

    Returns:
        iou value
    """
    _, iou_metric = get_metrics()
    model.eval()
    post_trans, post_label = get_post_trans()
    with eval_mode(model):
        for _, data in enumerate(dataloader):
            im, label = data["image"].to(device), data["label"].to(device)
            inferred = infer_seg(im, model, post_trans)
            test_labels = [post_label(i) for i in decollate_batch(label)]
            iou_metric(y_pred=inferred, y=test_labels)
        iou = iou_metric.aggregate().item()
        print(f"Iou is {iou} on {phase} set.")
        iou_metric.reset()
        return iou


class IoUMetric(CumulativeIterationMetric):
    """
    Compute IoU metric (Jaccard Score) between two tensors. It can support both multi-classes and
    multi-labels tasks. Input `y_pred` is compared with ground truth `y`. `y_pred` is expected to
    have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` to exclude the first category
    (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal
    from the background.
    `y_pred` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    Args:
        include_background: whether to skip IoU score computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan`
            values, available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``,
            ``"sum_batch"``, ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none",
            will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns
            (metric, not_nans). Here `not_nans` count the number of not nans for the metric, thus its
            shape equals to the shape of the metric.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.
    """

    def __init__(
        self,
        include_background: bool = True,
        reduction=MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
                The values should be binarized.
            y: ground truth to compute IoU metric. It must be one-hot format and first dim is batch.
                The values should be binarized.
        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(
                f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}."
            )
        # compute IoU score (BxC) for each channel for each batch
        return compute_iou(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            ignore_empty=self.ignore_empty,
        )

    def aggregate(self, reduction=None):  # type: ignore
        """
        Execute reduction logic for the output of `compute_iou`.
        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction
                       on `not-nan` values,  available reduction modes: {``"none"``, ``"mean"``,
                       ``"sum"``, ``"mean_batch"``, ``"sum_batch"``, ``"mean_channel"``,
                       ``"sum_channel"``}, default to `self.reduction`. if "none", will not
                       do reduction.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_iou(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    ignore_empty: bool = True,
) -> torch.Tensor:
    """Computes IoU score metric from full size Tensor and collects average.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
            The values should be binarized.
        y: ground truth to compute IoU metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip IoU score computation on the first channel of
            the predicted output. Defaults to True.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.
    Returns:
        IoU scores per batch and per class, (shape [batch_size, num_classes]).
    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o - intersection

    if ignore_empty is True:
        return torch.where(
            y_o > 0,
            (intersection) / denominator,
            torch.tensor(float("nan"), device=y_o.device),
        )
    return torch.where(
        denominator > 0,
        (intersection) / denominator,
        torch.tensor(1.0, device=y_o.device),
    )
