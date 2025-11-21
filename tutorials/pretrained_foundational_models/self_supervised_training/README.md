# Self-Supervised Contrastive Learning for Surgical videos
The focus of this tutorial is to walkthrough the process of doing Self-Supervised Learning using Contrastive Pre-training on Surgical Video data. 
As part of the walk-through we will guide through the steps needed to pre-process and extract the frames from the public *Cholec80 Dataset*. This will be required to run the tutorial.


The repository is organized as follows - 
* `Contrastive_learning_Notebook.ipynb` walks through the process of SSL in a tutorial style
* `train_simclr_multiGPU.py` enables running of "pre-training" on surgical data across multiple GPUs through the CLI
* `downstream_task_tool_segmentation.py` shows the process of "fine-tuning" for a downstream task starting from a pretrained checkpoint using [MONAI](https://github.com/Project-MONAI/MONAI)


## Dataset
To run through the full tutorial, it is required that the user downloads [Cholec80](http://camma.u-strasbg.fr/datasets) dataset. Additional preprocessing of the videos to extract individual frames can be performed using the python helper file as follows:

`python extract_frames.py --datadir <path_to_cholec80_dataset>` 

### Adapt to your own dataset
To run this with your own dataset, you will need to extract the frames and modify the `Pytorch Dataset/Dataloader` accordingly. For SSL pre-training, a really simple CSV file formatted as follows can be used. 
```
<path_to_frame>,<label>
```
where `<label>` can be a class/score for a downstream task, and is NOT used during pre-training.

```
# Snippet of csv file
/workspace/data/cholec80/frames/train/video01/1.jpg,0
/workspace/data/cholec80/frames/train/video01/2.jpg,0
/workspace/data/cholec80/frames/train/video01/3.jpg,0
/workspace/data/cholec80/frames/train/video01/4.jpg,0
/workspace/data/cholec80/frames/train/video01/5.jpg,0
....
```

## Environment
All environment/dependencies are captured in the [Dockerfile](Docker/Dockerfile). The exact software within the base container are [described here](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

### Create Docker Image/Container

```bash
DATA_DIR="/mnt/sdb/data"  # location of Cholec80 dataset
docker build -t surg_video_ssl_2202:latest Docker/

# sample Docker command (may need to update based on local setup)
docker run -it --gpus="device=1" \
    --name=SURGSSL_EXPS \
    -v $DATA_DIR:/workspace/data \
    -v `pwd`:/workspace/codes -w=/workspace/codes/ \
    -p 8888:8888 \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    surg_video_ssl_2202 jupyter lab
```

For environment dependencies refer to the [Dockerfile](Docker/Dockerfile)

## Launch Training

PRE-TRAINING
```bash
# Training on single GPU with `efficientnet_b0` backbone
python3 train_simclr_multigpu.py --gpus 1 --backbone efficientnet_b0 --batch_size 64

# Training on single GPU with `resnet50` backbone
python3 train_simclr_multigpu.py --gpus 4 --backbone resnet50 --batch_size 128
```

DOWNSTREAM TASK - Segmentation
This script shows an example of taking the checkpoint above and integrating it into [MONAI](https://project-monai.github.io).

```bash
# Fine-Tuning on "GPU 1" with 10% of the dataset, while freezing the encoder
python3 downstream_task_tool_segmentation.py --gpu 1 --perc 10 --exp simclr --freeze
```

## Model/Checkpoints information

As part of this tutorial, we are also releasing a few different checkpoints for users. These are detailed below. 

> **NOTE** : These checkpoints were trained using an internal dataset of Chelecystectomy videos provided by [Activ Surgical](https://www.activsurgical.com/) and NOT the Cholec80 dataset. 

### [Pre-Trained Backbones](https://drive.google.com/drive/folders/1NIfjydQ-o6Jl-DSAvTy5obw5XQ89Xje0?usp=share_link)
* ResNet18        - [link](https://drive.google.com/file/d/17w_LEI36JrHcUf5fGEufpyZaIk3Fp1Co/view?usp=sharing)
* ResNet50        - [link](https://drive.google.com/file/d/1fK87Nxit5bokYuMCbMG1cEHmEZUGBDlA/view?usp=share_link)
* efficientnet_b0 - [link](https://drive.google.com/file/d/1rgolweQ5HU6Kvf93jqLkaLVKE8DD_Aco/view?usp=sharing)

### Tool Segmentation Model
* MONAI - FlexibleUNet (efficientnet_b0) - [link](https://drive.google.com/file/d/1HLyccYY0AtZy8Sr1ty-gid-Fee4DVjWM/view?usp=share_link)

## Holoscan SDK
This tool Segmentation Model can be used to build a Holoscan App, using the process under section "Bring your own Model" within the [Holoscan SDK User guide](https://developer.download.nvidia.com/assets/Clara/ClaraHoloscan-1.pdf?t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1jb3Vyc2VzLm52aWRpYS5jb20vIiwibmNpZCI6InNvLW52c2gtODA1ODY2LXZ0MTIifQ==).


## Resources

[1] Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020).
A simple framework for contrastive learning of visual representations.
In International conference on machine learning (pp.
1597-1607).
PMLR.
([link](https://arxiv.org/abs/2002.05709))

[2] Chen, T., Kornblith, S., Swersky, K., Norouzi, M., and Hinton, G. (2020).
Big self-supervised models are strong semi-supervised learners.
NeurIPS 2021 ([link](https://arxiv.org/abs/2006.10029)).

[3] [Pytorch Lightning SSL Tutorial](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/13-contrastive-learning.html) | [Github](https://www.github.com/PytorchLightning/pytorch-lightning/)

[4] Ramesh, S., Srivastav, V., Alapatt, D., Yu, T., Muarli, A., et. al. (2023).   
Dissecting Self-Supervised Learning Methods for Surgical Computer Vision.
arXiv preprint arXiv:2207.00449.
([link](https://arxiv.org/abs/2207.00449))
