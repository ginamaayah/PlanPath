import os
import random

import numpy as np
import yaml


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = os.environ.get("CHECKPOINTS_ROOT")
    if root:
        cfg["paths"]["checkpoints_root"] = root
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def checkpoint_dir(cfg, stage):
    return os.path.join(cfg["paths"]["checkpoints_root"], stage)


def model_paths(cfg):
    return {
        "base": cfg["model"]["base"],
        "cpt": os.path.join(checkpoint_dir(cfg, "cpt"), "final"),
        "sft": os.path.join(checkpoint_dir(cfg, "sft"), "final"),
    }
