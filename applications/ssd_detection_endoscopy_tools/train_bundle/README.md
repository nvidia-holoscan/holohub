## Prepare SSD training pipeline with MONAI Bundle

### Install MONAI Dependencies

```Bash
docker pull projectmonai/monai:latest
```

### Collect Necessary Files

The original pipeline is in: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD and the training logic is defined in [main.py](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/main.py).
All necessary components are defined in [ssd/](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD/ssd), Therefore, we can put them into a bundle's `scripts/` folder. In addition, we also need to install all dependencies:

```Bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cp -r DeepLearningExamples/PyTorch/Detection/SSD/ssd/ scripts/
pip install -r DeepLearningExamples/PyTorch/Detection/SSD/requirements.txt
```

### Prepare Customized Workflow

Since the original training pipeline is using pythonic way, we can extend the [`BundleWorkflow` class of MONAI Bundle](https://github.com/Project-MONAI/MONAI/blob/dev/monai/bundle/workflows.py) and convert the content of `main.py` into it.
As shown in `scripts/workflow.py`, except of defining necessary functions (like `_set_property`, `_get_property`) that the `BundleWorkflow` requires, other parts (especially the train function) follow the original format. Please note that some of the logics are removed (such as distributed training) to simplify this example but you can still include them if needed.
As for the arguments, we prepared a config file in: `config/hyper_parameters.yaml` to store all of them, and users can still override them in CLI.

### Prepare Dataset

Please follow 

### Training Command

```Bash
python -m monai.bundle run_workflow "scripts.workflow.SSDWorkflow" --config_file configs/hyper_parameters.yaml --data <dataset to be trained>
```

### Evaluation Command

```Bash
python -m monai.bundle run_workflow "scripts.workflow.SSDWorkflow" --config_file configs/hyper_parameters.yaml --data <dataset to be trained> --checkpoint <checkpoint path> --mode evaluation
```

