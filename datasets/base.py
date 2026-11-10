from typing import Dict
from torch.utils.data import Dataset
from utils.registry import Registry
DATASET = Registry('Dataset')

def create_dataset(cfg: dict, phase: str, slurm: bool, **kwargs: Dict) -> Dataset:
    return DATASET.get(cfg.name)(cfg, phase, slurm, **kwargs)
