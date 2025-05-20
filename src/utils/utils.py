import os
from random import random

import numpy as np
import yaml

import torch

def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.use_deterministic_algorithms(True)  # For newer PyTorch versions
    os.environ['PYTHONHASHSEED'] = str(seed)

def use_best_hyperparams(args, dataset_name):
    best_params_file_path = "best_hyperparams.yml"
    os.chdir("..")      # Qin
    with open(best_params_file_path, "r") as file:
        hyperparams = yaml.safe_load(file)

    for name, value in hyperparams[dataset_name].items():
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            raise ValueError(f"Trying to set non existing parameter: {name}")

    return args


def get_available_accelerator():
    if torch.cuda.is_available():
        return "gpu"
    # Keep the following commented out as some of the operations
    # we use are currently not supported by mps
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return "mps"
    else:
        return "cpu"
