"""Generate grasps from a pretrained DiffMorphGrasp checkpoint."""

import os
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from functools import partial
from utils.misc import timestamp_str, compute_model_dim
from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
from models.visualizer import create_visualizer

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """Load checkpoint weights, accepting DataParallel key prefixes."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    saved_state_dict = torch.load(path, map_location=model.device, weights_only=True)['model']
    saved_state_dict = {(key[7:] if key.startswith('module.') else key): value
                        for key, value in saved_state_dict.items()}
    model.load_state_dict(saved_state_dict, strict=True)

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    if cfg.gpu is None or not torch.cuda.is_available():
        raise RuntimeError('Sampling requires a CUDA GPU for PointTransformer.')
    ckpt_path = cfg.checkpoint
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    torch.cuda.set_device(int(cfg.gpu))
    device = f'cuda:{cfg.gpu}'

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg.model.d_x = compute_model_dim(cfg.task)
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True

    vis_dir = os.path.join(cfg.exp_dir, 'samples', timestamp_str())
    os.makedirs(vis_dir, exist_ok=True)
    logger.add(os.path.join(vis_dir, 'sample.log'))
    logger.info('Configuration:\n' + OmegaConf.to_yaml(cfg))

    robot_names = cfg.robot_name

    test_dataloaders = {}
    total_test_size = 0

    for robot_name in robot_names:
        temp_cfg = cfg.task.dataset.copy()
        temp_cfg.robot_name = robot_name

        dataset = create_dataset(temp_cfg, 'test', cfg.slurm, case_only=True)
        total_test_size += len(dataset)
        logger.info(f'{robot_name}: {len(dataset)} test records')

        if cfg.model.scene_model.name == 'PointTransformer':
            collate_fn = partial(collate_fn_squeeze_pcd_batch, use_llm=cfg.model.use_llm)
        else:
            collate_fn = partial(collate_fn_general, use_llm=cfg.model.use_llm)

        dataloader = dataset.get_dataloader(
            batch_size=cfg.task.test.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.test.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        test_dataloaders[robot_name] = dataloader

    logger.info(f'Total test records: {total_test_size}')

    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)
    load_ckpt(model, path=ckpt_path)
    logger.info(f'Loaded checkpoint: {ckpt_path}')
    sampler = 'DPM-Solver++' if cfg.diffuser.sample.use_dpmsolver else 'DDPM'
    logger.info(f'Sampler: {sampler}; output: {vis_dir}')
    visualizer = create_visualizer(cfg.task.visualizer)
    visualizer.visualize(model, test_dataloaders, vis_dir)
    logger.info(f'Sampling complete: {vis_dir}')

if __name__ == '__main__':
    main()
