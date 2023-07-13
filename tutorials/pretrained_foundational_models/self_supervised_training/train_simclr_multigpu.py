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
---
"""

import argparse
import csv
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data Augemnation/Transforms


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8
        ),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class SurgicalVisionDataset_json(data.Dataset):
    def __init__(
        self, csv_path, transform=None, contrastive_training=True, train_set=True, loader=pil_loader
    ):
        with open(csv_path, "r") as csv_file:
            self.img_labels = list(csv.reader(csv_file, delimiter=","))
        self.transform = transform
        self.loader = loader
        self.return_labels = not contrastive_training

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = self.img_labels[index][0]
        image = self.loader(img_path)
        #         labels_phase = self.file_labels_phase[index]
        if self.transform is not None:
            image = self.transform(image)

        if self.return_labels:
            label = self.img_labels[index][1]
            return image, label

        return image, index


class SimCLR(pl.LightningModule):
    def __init__(
        self, hidden_dim, lr, temperature, weight_decay, max_epochs=500, backbone="resnet18"
    ):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        print(f"Loading {backbone} from torchvision...")
        # Base model f(.)
        self.pretrained = False

        print(f"Training for arch: {backbone}")

        if backbone == "resnet18":
            # if using ImageNet pretrained must redo fc layer first
            self.convnet = torchvision.models.resnet18(pretrained=self.pretrained)
            self.convnet.fc = nn.Linear(self.convnet.fc.in_features, 4 * hidden_dim)
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                nn.ReLU(inplace=True),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            # print(self.convnet.fc)

        elif backbone == "resnet50":
            self.convnet = torchvision.models.resnet50(pretrained=self.pretrained)
            self.convnet.fc = nn.Linear(self.convnet.fc.in_features, 4 * hidden_dim)
            self.convnet.fc = nn.Sequential(
                self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
                nn.ReLU(inplace=True),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )

        elif backbone == "efficientnet_b0":
            self.convnet = torchvision.models.efficientnet_b0(pretrained=self.pretrained)
            num_ftrs = self.convnet.classifier[1].in_features
            self.convnet.classifier[1] = nn.Linear(num_ftrs, 4 * hidden_dim)
            # num_classes is the output size of the last linear layer
            # The MLP for g(.) consists of Linear->ReLU->Linear
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,  # Linear(ResNet output, 4*hidden_dim)
                nn.ReLU(inplace=True),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )

        elif backbone == "efficientnet_b2":
            self.convnet = torchvision.models.efficientnet_b2(pretrained=self.pretrained)
            num_ftrs = self.convnet.classifier[1].in_features
            self.convnet.classifier[1] = nn.Linear(num_ftrs, 4 * hidden_dim)
            # num_classes is the output size of the last linear layer
            # The MLP for g(.) consists of Linear->ReLU->Linear
            self.convnet.classifier = nn.Sequential(
                self.convnet.classifier,  # Linear(ResNet output, 4*hidden_dim)
                nn.ReLU(inplace=True),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )

        else:
            raise ValueError("Model Arch {} not supported currently".format(backbone))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll, sync_dist=True)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],
                cos_sim.masked_fill(pos_mask, -9e15),
            ],  # First position positive example
            dim=-1,
        )
        comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        # self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), sync_dist=True)
        # self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), sync_dist=True)
        # self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=True)

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


def train_simclr(batch_size, max_epochs=300, num_GPUS=1, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR_MultiGPU"),
        accelerator="gpu",
        devices=num_GPUS,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_loss"),
            LearningRateMonitor("epoch"),
        ],
        strategy="ddp_find_unused_parameters_false",
    )
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SurgicalVision_SimCLR_ResNet.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        dataset_size = len(unlabeled_data)
        dataset_indices = list(range(dataset_size))
        np.random.seed(42)
        np.random.shuffle(dataset_indices)
        val_split_index = int(np.floor(0.2 * dataset_size))
        train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
        train_sampler = data.SubsetRandomSampler(train_idx)
        val_sampler = data.SubsetRandomSampler(val_idx)

        train_loader = data.DataLoader(
            unlabeled_data,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            sampler=train_sampler,
        )
        val_loader = data.DataLoader(
            unlabeled_data,
            batch_size=batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            sampler=val_sampler,
        )
        pl.seed_everything(42)  # To be reproducible
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, help="Batch Size of training run")
    parser.add_argument(
        "--gpus",
        type=int,
        choices=[1, 2, 3, 4],
        help="Number of GPUs to use in training",
        default=4,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs you'd like to train", default=300
    )
    parser.add_argument(
        "--backbone", type=str, help="Model architecture you wish to train", default="resnet18"
    )
    # parser.add_argument("--resume_training", type=str, help="Location of previously trained model",
    # default=False)

    args = vars(parser.parse_args())

    print(f'Starting SimCLR training for: {args["backbone"]}')

    # launch training
    train_simclr(
        batch_size=args["batch_size"],
        hidden_dim=128,
        lr=5e-4 * args["gpus"],
        temperature=0.07,
        weight_decay=1e-4,
        max_epochs=args["epochs"],
        num_GPUS=args["gpus"],
        backbone=args["backbone"],
    )


if __name__ == "__main__":
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "/home/clara/activ/data/"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "/home/clara/activ/models/test_checkpoints"
    # Select all CPU cores for I/O Data Preprocessing
    NUM_WORKERS = os.cpu_count()

    unlabeled_data = SurgicalVisionDataset_json(
        csv_path=DATASET_PATH + "activ.csv",
        transform=ContrastiveTransformations(contrast_transforms, n_views=2),
    )

    main()
