import os
import time

import numpy as np
import torch
from monai.bundle import BundleWorkflow, ConfigParser
from monai.utils import BundleProperty

from torch.optim.lr_scheduler import MultiStepLR

from monai.bundle import ConfigParser
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import ssd
from ssd.model import SSD300, ResNet, Loss
from ssd.utils import dboxes300_coco, Encoder
from ssd.logger import Logger
from ssd.evaluate import evaluate
from ssd.train import train_loop, tencent_trick, load_checkpoint
from ssd.data import get_coco_ground_truth

def generate_mean_std():
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean_val).cuda()
    std = torch.tensor(std_val).cuda()

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    return mean, std


class SSDWorkflow(BundleWorkflow):

    def __init__(self, config_file=None, workflow_type="train"):

        parser = ConfigParser()
        parser.read_config(config_file)

        super().__init__(workflow_type=workflow_type)
        self._props = {}
        self._set_props = {}
        self.parser = parser

        self.add_property("network", required=True)
        self.add_property("train_loader", required=True)
        self.add_property("val_dataset", required=True)
        self.add_property("val_loader", required=True)
        self.evaluator = None

    def _set_property(self, name, property, value):
        # stores user-reset initialized objects that should not be re-initialized.
        self._set_props[name] = value

    def _get_property(self, name, property):
        """
        The customized bundle workflow must implement required properties in:
        https://github.com/Project-MONAI/MONAI/blob/dev/monai/bundle/properties.py.
        """
        if name in self._set_props:
            self._props[name] = self._set_props[name]
            return self._props[name]
        if name in self._props:
            return self._props[name]
        try:
            value = getattr(self, f"get_{name}")()
        except AttributeError as err:
            if property[BundleProperty.REQUIRED]:
                raise ValueError(
                    f"Property '{name}' is required by the bundle format, "
                    f"but the method 'get_{name}' is not implemented."
                ) from err
            raise AttributeError from err
        self._props[name] = value
        return value

    def config(self, name, default="null", **kwargs):
        """read the parsed content (evaluate the expression) from the config file."""
        if default != "null":
            return self.parser.get_parsed_content(name, default=default, **kwargs)
        return self.parser.get_parsed_content(name, **kwargs)

    def initialize(self):
        pass

    def run(self):
        if str(self.workflow_type).startswith("train"):
            return self.train()
        if str(self.workflow_type).startswith("infer"):
            return self.infer()
        return self.validate()

    def finalize(self):
        pass

    def get_train_loader(self, local_seed):
        return ssd.data.get_train_loader(self.parser, local_seed)

    def get_val_loader(self, dataset):
        return ssd.data.get_val_dataloader(dataset, self.parser)

    def get_val_dataset(self):
        return ssd.data.get_val_dataset(self.parser)

    def get_network(self):
        network = SSD300(backbone=ResNet(backbone=self.parser.backbone,
                                    backbone_path=self.parser.backbone_path,
                                    weights=self.parser.torchvision_weights_version))
        
        return network

    def train(self):

        logger = Logger('Training logger', log_interval=self.parser.log_interval,
                                json_output=self.parser.json_summary)
        seed = self.parser.seed
        torch.manual_seed(seed)
        np.random.seed(seed=seed)

        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)
        cocoGt = get_coco_ground_truth(self.parser)
        train_loader = self.get_train_loader(seed - 2**31)
        val_dataset = self.get_val_dataset()
        val_dataloader = self.get_val_loader(val_dataset)
        ssd300 = self.get_network()
        learning_rate = self.parser.learning_rate * self.parser.N_gpu * (self.parser.batch_size / 32)
        start_epoch = 0
        iteration = 0
        loss_func = Loss(dboxes)
        use_cuda = self.parser.use_cuda
        if use_cuda:
            ssd300.cuda()
            loss_func.cuda()

        optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=learning_rate,
                                    momentum=self.parser.momentum, weight_decay=self.parser.weight_decay)
        scheduler = MultiStepLR(optimizer=optimizer, milestones=self.parser.multistep, gamma=0.1)

        if self.parser.checkpoint is not None:
            if os.path.isfile(self.parser.checkpoint):
                load_checkpoint(ssd300, self.parser.checkpoint)
                checkpoint = torch.load(self.parser.checkpoint, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
                start_epoch = checkpoint['epoch']
                iteration = checkpoint['iteration']
                scheduler.load_state_dict(checkpoint['scheduler'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('Provided checkpoint is not path to a file')
                return

        inv_map = {v: k for k, v in val_dataset.label_map.items()}

        total_time = 0

        if self.parser.mode == 'evaluation':
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, self.parser)
            if self.parser.local_rank == 0:
                print('Model precision {} mAP'.format(acc))
            return

        scaler = torch.cuda.amp.GradScaler(enabled=self.parser.amp)
        mean, std = generate_mean_std()

        for epoch in range(start_epoch, self.parser.epochs):
            start_epoch_time = time.time()
            iteration = train_loop(ssd300, loss_func, scaler,
                                        epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                        logger, self.parser, mean, std)
            if self.parser.mode in ["training"]:
                scheduler.step()
            end_epoch_time = time.time() - start_epoch_time
            total_time += end_epoch_time

            if self.parser.local_rank == 0:
                logger.update_epoch_time(epoch, end_epoch_time)

            if epoch in self.parser.evaluation:
                acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, self.parser)

                if self.parser.local_rank == 0:
                    logger.update_epoch(epoch, acc)

            if self.parser.save and self.parser.local_rank == 0:
                print("saving model...")
                obj = {'epoch': epoch + 1,
                    'iteration': iteration,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'label_map': val_dataset.label_info}
                obj['model'] = ssd300.state_dict()
                os.makedirs(self.parser.save, exist_ok=True)
                save_path = os.path.join(self.parser.save, f'epoch_{epoch}.pt')
                torch.save(obj, save_path)
                logger.log('model path', save_path)
            train_loader.reset()
        logger.log_summary()
