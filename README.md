# Repository for the hackathon of Deep Learning course, a.y. 2024-2025

## Authors: Matteo Carnebella, Francesco Di Lorenzo

## Usage

```bash
python main.py --test_path /path/to/test/dataset [OPTIONS]
```

### Required Arguments
- `--test_path`: Path to the test dataset

### Optional Arguments
- `--train_path`: Path to the training dataset
- `--gnn`: GNN architecture (default: `def_gcn`)
  - Options: `def_gcn`, `def_gin`, `simple_gin`, `simple_gine`
- `--criterion`: Loss function (default: `ce`)
  - Options: `ce`, `focal`, `gcod`
- `--u_lr`: Learning rate for u parameters (default: `1.0`)
- `--batch_size`: Input batch size for training (default: `32`)
- `--epochs`: Number of epochs to train (default: `1000`)
- `--early_stopping`: Enable early stopping (default: `False`)
- `--patience`: Max number of epochs without training improvements (default: `50`)
- `--train_val_split`: Enable 80% train 20% validation split (default: `False`)

### Example Usage

```bash
# Basic usage with required argument
python main.py --test_path <path_to_test.json.gz>

# Training with custom GNN and parameters
python main.py --train_path <path_to_train.json.gz> --test_path .<path_to_test.json.gz> --gnn def_gin --epochs 500 --batch_size 64

# Training with early stopping and validation split
python main.py --train_path <path_to_train.json.gz> --test_path <path_to_test.json.gz> --early_stopping --patience 30 --train_val_split
```

## Overview of the Method

This repository contains the code used to solve the Deep Learning course's hackathon. The available models are:
- a GNN using pytorch geometric GIN module
- a GNN using pytorch geometric GINE module
Three loss options are available:
- Cross Entropy Loss
- Focal Loss
- GCOD Loss, as reported in the following paper: [Robustness of Graph Classification: failure modes, causes, and
noise-resistant loss in Graph Neural Networks](https://arxiv.org/pdf/2412.08419)

Using the optional arguments, one can launch the script specifying the parameters described above.  
Checkpoints are saved every patience/5 epochs (if num_epochs is not passed as argument, the default value of 50 will be used) in the `/checkpoints/<dataset_name>/`  
Plots of the train (and validation, if the train_val_split is enabled) loss, accuracy and f1-score are saved in `/logs/<dataset_name>/`  
The auxiliary code is saved in `/src`  
The results of the runs are saved in `/submission`