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

The original training pipeline of SSD is run in command line, and can parse arguments into the `main.py` file.
In order to convert into a MONAI Bundle format, we can extend the [`PythonicWorkflow` class of MONAI Bundle](https://github.com/Project-MONAI/MONAI/blob/dev/monai/bundle/workflows.py) and convert the content of `main.py` into it.

As for the arguments, we can prepare a config file (MONAI Bundle workflow can parse it) to replace it (see `configs/hyper_parameters.yaml`), and users can still override them in CLI.

As for other functions, we don't need to modify them.

As for the code under `if __name__ == "__main__":` which will be executed directly when run the script, we only need a bit modification and put them into `run` function (see `def run()`)

### Prepare Metadata Config

`configs/metadata.json` is an import part of a MONAI Bundle. It contains data like the bundle version, change log, dependencies, and the information of the network input and output.
It helps authors and users to manage versions, distribute and reproduce results.

### Prepare Dataset

Please follow 

### Training Command

```Bash
python -m monai.bundle run_workflow "scripts.workflow.SSDWorkflow" --config_file configs/hyper_parameters.yaml --data <dataset to be trained> --save <folder to save weights> --json_summary <json file to save logs>
```

### Evaluation Command

```Bash
python -m monai.bundle run_workflow "scripts.workflow.SSDWorkflow" --config_file configs/hyper_parameters.yaml --data <dataset to be trained> --checkpoint <checkpoint path> --mode evaluation
```

