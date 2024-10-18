import os
import time
import sys

import torch
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data.distributed
from monai.bundle import PythonicWorkflow

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import ssd
from ssd.model import SSD300, ResNet, Loss
from ssd.utils import dboxes300_coco, Encoder
from ssd.logger import Logger, BenchLogger
from ssd.evaluate import evaluate
from ssd.train import train_loop, tencent_trick, load_checkpoint, benchmark_train_loop, benchmark_inference_loop
from ssd.data import get_train_loader, get_val_dataset, get_val_dataloader, get_coco_ground_truth

import dllogger as DLLogger

# Apex imports
try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")


def generate_mean_std(args):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean_val).cuda()
    std = torch.tensor(std_val).cuda()

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    return mean, std

def train(train_loop_func, logger, args):
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda

    # Setup multi-GPU if necessary
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.N_gpu = torch.distributed.get_world_size()
    else:
        args.N_gpu = 1

    if args.seed is None:
        args.seed = np.random.randint(1e4)

    if args.distributed:
        args.seed = (args.seed + torch.distributed.get_rank()) % 2**32
    print("Using seed = {}".format(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)


    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    train_loader = get_train_loader(args, args.seed - 2**31)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    ssd300 = SSD300(backbone=ResNet(backbone=args.backbone,
                                    backbone_path=args.backbone_path,
                                    weights=args.torchvision_weights_version))
    args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
    start_epoch = 0
    iteration = 0
    loss_func = Loss(dboxes)

    if use_cuda:
        ssd300.cuda()
        loss_func.cuda()

    optimizer = torch.optim.SGD(tencent_trick(ssd300), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    if args.distributed:
        ssd300 = DDP(ssd300)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            load_checkpoint(ssd300.module if args.distributed else ssd300, args.checkpoint)
            checkpoint = torch.load(args.checkpoint,
                                    map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()))
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            scheduler.load_state_dict(checkpoint['scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Provided checkpoint is not path to a file')
            return

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    total_time = 0

    if args.mode == 'evaluation':
        acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
        if args.local_rank == 0:
            print('Model precision {} mAP'.format(acc))
        return

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    mean, std = generate_mean_std(args)

    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        iteration = train_loop_func(ssd300, loss_func, scaler,
                                    epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                    logger, args, mean, std)
        if args.mode in ["training", "benchmark-training"]:
            scheduler.step()
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        if args.local_rank == 0:
            logger.update_epoch_time(epoch, end_epoch_time)

        if epoch in args.evaluation:
            acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

            if args.local_rank == 0:
                logger.update_epoch(epoch, acc)

        if args.save and args.local_rank == 0:
            print("saving model...")
            obj = {'epoch': epoch + 1,
                   'iteration': iteration,
                   'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(),
                   'label_map': val_dataset.label_info}
            if args.distributed:
                obj['model'] = ssd300.module.state_dict()
            else:
                obj['model'] = ssd300.state_dict()
            os.makedirs(args.save, exist_ok=True)
            save_path = os.path.join(args.save, f'epoch_{epoch}.pt')
            torch.save(obj, save_path)
            logger.log('model path', save_path)
        train_loader.reset()
    DLLogger.log((), { 'total time': total_time })
    logger.log_summary()


def log_params(logger, args):
    logger.log_params({
        "dataset path": args.data,
        "epochs": args.epochs,
        "batch size": args.batch_size,
        "eval batch size": args.eval_batch_size,
        "no cuda": args.no_cuda,
        "seed": args.seed,
        "checkpoint path": args.checkpoint,
        "mode": args.mode,
        "eval on epochs": args.evaluation,
        "lr decay epochs": args.multistep,
        "learning rate": args.learning_rate,
        "momentum": args.momentum,
        "weight decay": args.weight_decay,
        "lr warmup": args.warmup,
        "backbone": args.backbone,
        "backbone path": args.backbone_path,
        "num workers": args.num_workers,
        "AMP": args.amp,
        "precision": 'amp' if args.amp else 'fp32',
    })


class SSDWorkflow(PythonicWorkflow):

    def __init__(self, workflow_type: str = "train", config_file: str | None = None, **override):
        super().__init__(workflow_type=workflow_type, config_file=config_file, **override)
        
    def run(self):
        args = self.parser
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        if args.local_rank == 0:
            os.makedirs('./models', exist_ok=True)

        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.benchmark = True

        # write json only on the main thread
        args.json_summary = args.json_summary if args.local_rank == 0 else None

        if args.mode == 'benchmark-training':
            train_loop_func = benchmark_train_loop
            logger = BenchLogger('Training benchmark', log_interval=args.log_interval,
                                json_output=args.json_summary)
            args.epochs = 1
        elif args.mode == 'benchmark-inference':
            train_loop_func = benchmark_inference_loop
            logger = BenchLogger('Inference benchmark', log_interval=args.log_interval,
                                json_output=args.json_summary)
            args.epochs = 1
        else:
            train_loop_func = train_loop
            logger = Logger('Training logger', log_interval=args.log_interval,
                            json_output=args.json_summary)

        log_params(logger, args)

        train(train_loop_func, logger, args)

    def finalize(self):
        pass
