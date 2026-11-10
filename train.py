import os
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from utils.misc import compute_model_dim
from utils.io import mkdir_if_not_exists
from utils.plot import Ploter
from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
from models.visualizer import create_visualizer
from tqdm import tqdm
from functools import partial
import random

def save_ckpt(model: torch.nn.Module, epoch: int, step: int, path: str, save_scene_model: bool, optimizer=None, cfg=None) -> None:
    saved_state_dict = {}
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        if 'scene_model' in key and not save_scene_model:
            continue

        saved_state_dict[key] = model_state_dict[key]

    logger.info('Saving checkpoint' + ('' if save_scene_model else ' (excluding scene encoder)'))
    checkpoint = {
        'model': saved_state_dict,
        'epoch': epoch, 'step': step,
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if cfg is not None:
        checkpoint['config'] = OmegaConf.to_container(cfg, resolve=True)
        from datasets.multidex_shadowhand_ur import MultiDexShadowHandUR
        checkpoint['train_objects'] = MultiDexShadowHandUR._train_split
        checkpoint['test_objects'] = MultiDexShadowHandUR._test_split
    torch.save(checkpoint, path)

def load_ckpt(model: torch.nn.Module, ckpt_dir: str, save_model_separately: bool, optimizer=None) -> (int, int): # type: ignore
    if not os.path.exists(ckpt_dir):
        return 0, 0
    if save_model_separately:
        checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith('model_') and f.endswith('.pth')]
        if not checkpoint_files:
            return 0, 0

        if 'model_latest.pth' in checkpoint_files:
            latest_ckpt = 'model_latest.pth'
        else:
            numeric_checkpoints = [f for f in checkpoint_files if f.split('_')[1].replace('.pth', '').isdigit()]
            if not numeric_checkpoints:
                return 0, 0
            latest_ckpt = max(numeric_checkpoints, key=lambda f: int(f.split('_')[1].replace('.pth', '')))

        ckpt_path = os.path.join(ckpt_dir, latest_ckpt)

    else:
        ckpt_path = os.path.join(ckpt_dir, 'model.pth')
        if not os.path.exists(ckpt_path):
            return 0, 0

    logger.info(f'Loading checkpoint from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=model.device, weights_only=True)
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    saved_state_dict = checkpoint['model']
    saved_state_dict = {(key[7:] if key.startswith('module.') else key): value
                        for key, value in saved_state_dict.items()}
    model.load_state_dict(saved_state_dict, strict=True)
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    elif optimizer is not None:
        logger.warning('Legacy checkpoint has no optimizer state; Adam starts fresh.')
    return epoch + 1, step

def train(cfg: DictConfig, writer: SummaryWriter) -> None:
    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'

    robot_names = cfg.robot_name

    train_dataloaders = {}
    total_train_size = 0

    for robot_name in robot_names:
        temp_cfg = cfg.task.dataset.copy()
        temp_cfg.robot_name = robot_name

        dataset = create_dataset(temp_cfg, 'train', cfg.slurm)
        total_train_size += len(dataset)
        logger.info(f'Load {robot_name} train dataset size: {len(dataset)}')

        if cfg.model.scene_model.name == 'PointTransformer':
            collate_fn = partial(collate_fn_squeeze_pcd_batch, use_llm=cfg.model.use_llm)
        else:
            collate_fn = partial(collate_fn_general, use_llm=cfg.model.use_llm)

        dataloader = dataset.get_dataloader(
            batch_size=cfg.task.train.batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.task.train.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        train_dataloaders[robot_name] = dataloader

    logger.info(f'Total train dataset size: {total_train_size}')

    if cfg.task.visualizer.visualize:
        temp_cfg = cfg.task.dataset.copy()
        temp_cfg.robot_name = robot_names[0]
        test_dataset = create_dataset(temp_cfg, 'test', cfg.slurm, case_only=True)

    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)

    params = []
    nparams = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())

    params_group = [
        {'params': params, 'lr': cfg.task.lr},
    ]
    optimizer = torch.optim.Adam(params_group)
    logger.info(f'{len(params)} parameters for optimization.')
    logger.info(f'total model size is {sum(nparams)}.')

    if cfg.task.visualizer.visualize:
        visualizer = create_visualizer(cfg.task.visualizer)

    start_epoch, start_step = load_ckpt(model, cfg.ckpt_dir, cfg.save_model_seperately, optimizer)

    step = start_step
    robot_restart_counts = {robot: 0 for robot in robot_names}

    for epoch in range(start_epoch, cfg.task.train.num_epochs):
        model.train()

        robot_usage_counts = {robot: 0 for robot in robot_names}

        iterators = {name: iter(dl) for name, dl in train_dataloaders.items()}

        robot_completed_rounds = {robot: 0 for robot in robot_names}

        total_batches = sum(len(dl) for dl in train_dataloaders.values())

        progress_bar = tqdm(range(total_batches), desc=f'Epoch {epoch + 1}/{cfg.task.train.num_epochs}')

        for it in progress_bar:
            current_robot = random.choice(robot_names)

            try:
                data = next(iterators[current_robot])

            except StopIteration:
                iterators[current_robot] = iter(train_dataloaders[current_robot])
                data = next(iterators[current_robot])
                robot_restart_counts[current_robot] += 1
                robot_completed_rounds[current_robot] += 1
                logger.info(f'{current_robot} dataloader restarted at iteration {it}, round {robot_completed_rounds[current_robot]}')

                if all(rounds >= 1 for rounds in robot_completed_rounds.values()):
                    logger.info(f'All robots completed at least one round, ending epoch at iteration {it}')
                    break

            robot_usage_counts[current_robot] += 1

            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            optimizer.zero_grad()
            data['epoch'] = epoch
            outputs = model(data)
            outputs['loss'].backward()
            optimizer.step()
            total_loss = outputs['loss'].item()

            progress_bar.set_postfix(loss=total_loss, robot=current_robot)

            if (step + 1) % cfg.task.train.log_step == 0:
                log_str = f'[TRAIN] ==> Epoch: {epoch+1:3d} | Iter: {it+1:5d} | Step: {step+1:7d} | Loss: {total_loss:.3f} | Robot: {current_robot}'
                logger.info(log_str)
                writer.add_scalar('Loss/train', total_loss, step)
                for key in outputs:
                    val = outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]
                    Ploter.write({
                        f'train/{key}': {'plot': True, 'value': val, 'step': step},
                        'train/epoch': {'plot': True, 'value': epoch, 'step': step},
                    })

            step += 1

        logger.info(f'Epoch {epoch+1} completed:')
        logger.info(f'Robot usage counts (this epoch): {robot_usage_counts}')
        logger.info(f'Robot restart counts (total): {robot_restart_counts}')
        logger.info(f'Robot completed rounds (this epoch): {robot_completed_rounds}')

        if (epoch + 1) % cfg.save_model_interval == 0:
            save_path = os.path.join(
                cfg.ckpt_dir,
                f'model_{epoch}.pth' if cfg.save_model_seperately else 'model.pth'
            )

            save_ckpt(
                model=model, epoch=epoch, step=step, path=save_path,
                save_scene_model=cfg.save_scene_model,
                optimizer=optimizer, cfg=cfg,
            )
        latest_path = os.path.join(cfg.ckpt_dir, 'model_latest.pth')
        save_ckpt(
            model=model, epoch=epoch, step=step, path=latest_path,
            save_scene_model=cfg.save_scene_model,
            optimizer=optimizer, cfg=cfg,
        )

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    cfg.model.d_x = compute_model_dim(cfg.task)
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True
        logger.remove(handler_id=0)

    logger.add(cfg.exp_dir + '/runtime.log')

    mkdir_if_not_exists(cfg.tb_dir)
    mkdir_if_not_exists(cfg.vis_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)

    writer = SummaryWriter(log_dir=cfg.tb_dir)
    Ploter.setWriter(writer)

    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
    logger.info('Begin training..')

    train(cfg, writer)

    writer.close()
    logger.info('End training..')

if __name__ == '__main__':
    main()
