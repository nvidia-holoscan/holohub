"""
SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
---

"""

import argparse
import os
import time

import monai
import torch
from monai.data import decollate_batch
from monai.losses import GeneralizedDiceFocalLoss
from monai.networks import eval_mode
from monai.networks.nets.flexible_unet import FlexibleUNet
from monai.networks.utils import copy_model_state
from monai.optimizers import generate_param_groups
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from utils import (
    check_set_and_save,
    get_dlds_dict,
    get_metrics,
    get_post_trans,
    get_test_metrics,
    infer_seg,
    save_training_curve,
)


def load_model(modelname, in_channels, backbone):
    """
    Loading a model by name.

    Args:
        modelname: a whole path name of the model that need to be loaded.
        in_channels: the number of input channel of the model.

    Returns:
        model in torch and a model open state.
    """
    isopen = os.path.exists(modelname)
    if not isopen:
        return None, isopen

    model = FlexibleUNet(in_channels=in_channels, out_channels=2, backbone=backbone).to(device)

    model.load_state_dict(torch.load(modelname, map_location=device))

    return model, isopen


def load_simclr_backbone(monai_model, simclr_path):
    monai_keys = [
        key
        for key in monai_model.state_dict().keys()
        if ((key.split(".")[0] == "encoder") & (key.split(".")[1] != "_fc"))
    ]
    # filter out MLP layer from SIMCLR
    simclr_checkpoint = torch.load(simclr_path, map_location=device)
    simclr_keys = [
        key
        for key in simclr_checkpoint["state_dict"].keys()
        if ((key.split(".")[0] == "convnet") & (key.split(".")[1] != "classifier"))
    ]
    mapping = dict(zip(simclr_keys, monai_keys))
    ## update in place
    model_a_b, updated_keys, unchanged_keys = copy_model_state(
        monai_model, simclr_checkpoint["state_dict"], mapping=mapping, inplace=True
    )
    return model_a_b, updated_keys, unchanged_keys


def train_network(dataloader_dict, dataset_dict, max_epochs, best_model_path, exp, freeze):
    """
    Training the model and evaluating it on the validation dataset
    each val interval.

    Args:
        dataloader_dict: dict that contains train, valid and test dataloaders.
        that can be found by a key of the same name.
        dataset_dict:dict that contains train, valid and test datasets.
        that can be found by a key of the same name.
        max_epochs: max epoch number for training.
        best_model_path: path name to save the best model.
    """
    save_path = os.path.dirname(best_model_path)
    log_path = os.path.join(save_path, "tb_logs")
    os.makedirs(log_path, exist_ok=True)
    summary_writer = SummaryWriter(log_path)
    # Create metrics, network, loss function optimizer, etc.
    dice_metric, iou_metric = get_metrics()

    post_trans, post_label = get_post_trans()
    in_channels = 3

    # model = RanzcrNetV2(

    # Update model to load from a saved checkpoint; uses MONAI function but
    # encoder weights replaced with simclr backone
    backbone = "efficientnet-b0"
    checkpoints = {
        "efficientnet-b0": "SurgicalVision_SimCLR_Effb0.ckpt",
    }

    model = FlexibleUNet(in_channels=in_channels, out_channels=2, backbone=backbone).to(device)
    print(f"### Using {exp} pre-trained backbone - {backbone} ###")

    if exp == "simclr":
        _, updated_keys, unchanged_keys = load_simclr_backbone(model, checkpoints[backbone])
        ## Updating optimizer to choose learning rate selectively
        # stop gradients for the pretrained weights
        if freeze:
            print(f"### Freezing {exp} pre-trained backbone - {backbone} ###")

            for x in model.named_parameters():
                if x[0] in updated_keys:
                    x[1].requires_grad = False
            optimizer = torch.optim.Adam(model.parameters(), 1e-5)

        else:
            params = generate_param_groups(
                network=model,
                layer_matches=[lambda x: x[0] in unchanged_keys],
                match_types=["filter"],
                lr_values=[1e-4],
                include_others=True,
            )
            optimizer = torch.optim.Adam(params, 1e-5)
    elif exp == "imagenet":
        optimizer = torch.optim.Adam(model.parameters(), 1e-4)  ## train all parameters
    loss_function = GeneralizedDiceFocalLoss(
        include_background=False, softmax=True, to_onehot_y=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=100, T_mult=1
    )

    # start a typical PyTorch training
    train_dl = dataloader_dict["train"]
    val_dl = dataloader_dict["valid"]
    test_dl = dataloader_dict["test"]
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    curve_data = dict()
    curve_data["train_loss"] = []
    curve_data["val_dice"] = []
    curve_data["val_iou"] = []
    curve_data["epoch"] = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0

        for batch_data in train_dl:
            # print("*** - batch data", batch_data["image"].shape)
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= len(train_dl)
        curve_data["train_loss"].append(epoch_loss)
        curve_data["epoch"].append(epoch + 1)
        summary_writer.add_scalar("train/loss", epoch_loss, epoch + 1)

        if (epoch + 1) % val_interval == 0:
            with eval_mode(model):
                val_outputs = None
                for val_data in val_dl:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(
                        device
                    )
                    val_outputs = infer_seg(val_images, model, post_trans)
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    iou_metric(y_pred=val_outputs, y=val_labels)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                iou = iou_metric.aggregate().item()
                iou_metric.reset()

                test_iou = get_test_metrics(test_dl, "test", model, device)

                curve_data["val_dice"].append(metric)
                curve_data["val_iou"].append(iou)
                print(
                    "Epoch",
                    epoch + 1,
                    time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime(time.time())),
                    "IOU:",
                    iou,
                )
                summary_writer.add_scalars(
                    "val/metrics",
                    {"iou": iou, "dice": metric, "test_iou": test_iou},
                    epoch + 1,
                )

                if iou > best_metric:
                    best_metric = iou
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), best_model_path)

    # Plotting stuff
    save_path = os.path.dirname(best_model_path)
    save_training_curve(curve_data, save_path)

    # print the information of the best model.
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

    # evaluation on test set
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    get_test_metrics(test_dl, "test", model, device)


def training_pipeline(
    data_path, save_path, model_name, max_epochs=100, train_perc=100, exp="simclr", freeze=False
):
    """
    Build up the training pipeline including a data io part, a data check part
    and a training part.

    Args:
        data_path:  the root path of dataset.
        save_path:  the path to save models and images produced during training.
        model_name: the name of the best model to be saved.
        max_epochs: the maximum epoch number for training default to 100.
    """
    # Set deterministic training for reproducibility
    set_determinism(seed=0)

    phase_list = ["train", "valid", "test"]
    # Making the save path and the modelname.
    if not os.path.exists(data_path):
        raise Exception("Path {} doesn't exists.".format(data_path))
    os.makedirs(save_path, exist_ok=True)
    save_model = os.path.join(save_path, model_name)

    # Generating datasets and dataloaders - select percentage of training data
    dataset_dict, dataloader_dict = get_dlds_dict(data_path, phase_list, train_perc=train_perc)

    # Picking and saving images randomly in the validation set.
    check_set_and_save("valid", dataset_dict["valid"], save_path)

    # Training
    train_network(dataloader_dict, dataset_dict, max_epochs, save_model, exp, freeze)


if __name__ == "__main__":
    monai.config.print_config()
    parser = argparse.ArgumentParser()
    # A path to save the result.
    parser.add_argument(
        "--savepath",
        type=str,
        default=r"/home/clara/models/toolseg/res",
        help="The path to save models and results.",
    )
    # A segmentation model name for saving.
    parser.add_argument(
        "--modelname",
        type=str,
        default=r"best_seg_model.pt",
        help="Input a segmentation model name for saving.",
    )
    # The path of dataset.
    parser.add_argument(
        "--datapath",
        type=str,
        default=r"/home/clara/data/tool_seg_dataset/",
        help="The root path of train, validation and test datasets.",
    )

    parser.add_argument(
        "--gpu", type=int, choices=[0, 1, 2, 3], help="Select GPU to use in training", default=0
    )

    # Max training epoch number.
    parser.add_argument(
        "--maxepoch", type=int, default=100, help="The max epoch number for training."
    )

    # Percent of data for fine-tuning
    parser.add_argument("--perc", type=int, default=100, help="Percent of data for fine-tuning.")

    # Percent of data for fine-tuning
    parser.add_argument(
        "--exp", type=str, help="simclr or imagenet", choices=["simclr", "imagenet"]
    )

    # Freeze pre-trained backbone for simclr
    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    save_path = args.savepath
    data_path = args.datapath
    max_epochs = args.maxepoch
    device = torch.device("cuda:" + str(args.gpu))
    print(f"Using device: {device}")

    train_perc = args.perc
    model_name = (
        lambda x: x.split(".")[0] + f"_{args.exp}" + f"_{train_perc}perc." + x.split(".")[1]
    )(args.modelname)
    training_pipeline(
        data_path,
        save_path,
        model_name,
        max_epochs=max_epochs,
        train_perc=train_perc,
        exp=args.exp,
        freeze=args.freeze,
    )
