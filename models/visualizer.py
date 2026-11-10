"""Generate one batch of grasps per object and hand."""

import os

import torch
from einops import rearrange
from tqdm import tqdm

from utils.myutils import ROBOT_JOINT_MAPPINGS, get_matrix, remap_joint_info
from utils.penetration import refine_grasps, should_optimize_penetration
from utils.registry import Registry
from utils.rot6d import random_rot, rot_to_orthod6d
from utils.retarget import align_inspire_root, align_leap_root, couple_inspire_joints
from pathlib import Path
import pytorch_kinematics as pk
VISUALIZER = Registry('Visualizer')


@VISUALIZER.register()
class GraspGenURVisualizer:
    def __init__(self, cfg):
        self.ksample = cfg.ksample
        self.masked_fingers = cfg.masked_fingers
        self.use_llm = cfg.use_llm
        self.objects = cfg.get('objects')
        self.asset_dir = cfg.asset_dir
        self.penetration_optimization = cfg.penetration_optimization

    def _get_padding_mask(self, robot, device):
        mask = torch.zeros(27, device=device)
        mask[:3] = 1
        for index in ROBOT_JOINT_MAPPINGS[robot]['joint_to_unified'].values():
            mask[3 + index] = 1
        return mask

    @torch.no_grad()
    def visualize(self, model, dataloaders, save_dir):
        model.eval()
        device = model.device
        if 'shadowhand' in dataloaders: # add joint limits for shadowhand
            chain = pk.build_chain_from_urdf(
                Path('data/urdf/shadowhand/shadow_hand_right_extended.urdf').read_bytes()
            )
            shadow_limits = tuple(torch.as_tensor(bound[6:], device=device) for bound in chain.get_joint_limits())
        objects = sorted({name for loader in dataloaders.values() for name in loader.dataset._test_split})
        if self.objects is not None:
            unknown = set(self.objects) - set(objects)
            if unknown:
                raise ValueError(f'Objects outside the evaluation split: {sorted(unknown)}')
            objects = [name for name in objects if name in self.objects]
        count = sum(sum(name in loader.dataset._test_split for name in objects) * self.ksample for loader in dataloaders.values())
        with tqdm(total=count, desc='Sampling') as progress:
            for object_name in objects:
                for robot, loader in dataloaders.items():
                    dataset = loader.dataset
                    if object_name not in dataset._test_split:
                        continue
                    cloud = torch.as_tensor(dataset.scene_pcds[object_name], device=device)
                    cloud = cloud[:dataset.num_points].unsqueeze(0).repeat(self.ksample, 1, 1)
                    rotation = torch.stack([random_rot(device) for _ in range(self.ksample)]).double()
                    points = (rotation @ cloud[:, :, :3].transpose(1, 2)).transpose(1, 2)
                    normals = (rotation @ cloud[:, :, 3:6].transpose(1, 2)).transpose(1, 2)
                    adjacency, _, parent, _, child, _ = get_matrix(robot, 'cpu', self.masked_fingers)
                    limits, origin, axis, box = remap_joint_info(robot, self.masked_fingers)
                    mask = self._get_padding_mask(robot, device)
                    text = dataset.scene_text[object_name] if self.use_llm else None
                    data = {
                        'x': torch.zeros(self.ksample, 27, device=device),
                        'pos': rearrange(points, 'b n c -> (b n) c'),
                        'normal': normals,
                        'feat': rearrange(normals, 'b n c -> (b n) c'),
                        'scene_rot_mat': rotation,
                        'scene_id': [object_name] * self.ksample,
                        'cam_trans': [None] * self.ksample,
                        'text': text * self.ksample if text is not None else None,
                        'sentence_cnt': [len(text)] * self.ksample if text is not None else None,
                        'padding_mask': mask.unsqueeze(0).repeat(self.ksample, 1),
                        'adjacency_matrix': adjacency.to(device),
                        'parent_matrix': parent.to(device),
                        'child_matrix': child.to(device),
                        'joint_limits': limits.unsqueeze(0).repeat(self.ksample, 1, 1).to(device),
                        'joint_origin': origin.unsqueeze(0).repeat(self.ksample, 1, 1).to(device),
                        'joint_axis': axis.unsqueeze(0).repeat(self.ksample, 1, 1).to(device),
                        'joint_box': box.unsqueeze(0).repeat(self.ksample, 1, 1).to(device),
                        'robot_name': [robot] * self.ksample,
                        'masked_fingers': [self.masked_fingers] * self.ksample,
                        'offset': torch.arange(1, self.ksample + 1, device=device, dtype=torch.int32) * points.shape[1],
                    }
                    outputs = model.sample(data, k=1).squeeze(1)[:, -1, :].double()
                    angles = dataset.angle_denormalize(outputs[:, 3:].cpu()).to(device) if dataset.normalize_x else outputs[:, 3:]
                    if robot in ("leaphand", "inspire"):
                        angles = angles.clamp(dataset._joint_angle_lower.to(device), dataset._joint_angle_upper.to(device))
                    translation = dataset.trans_denormalize(outputs[:, :3].cpu()).to(device) if dataset.normalize_x_trans else outputs[:, :3]
                    translation = (rotation.transpose(1, 2) @ translation.unsqueeze(-1)).squeeze(-1)
                    rot6d = rot_to_orthod6d(rotation.transpose(1, 2))
                    if robot == 'shadowhand': # clamp shadowhand joint angles
                        angles = angles.clamp(*shadow_limits)
                    poses = torch.cat((translation, rot6d, angles), dim=-1).float().cpu()
                    if robot == "leaphand":
                        poses = align_leap_root(poses)
                    elif robot == "inspire":
                        poses = align_inspire_root(poses)
                        poses[:, 9:] = couple_inspire_joints(poses[:, 9:])
                    penetration_before = penetration_after = None
                    if should_optimize_penetration(robot, self.penetration_optimization):
                        with torch.enable_grad():
                            refined, penetration_before, penetration_after = refine_grasps(
                                poses.to(device), object_name, self.asset_dir, robot)
                            if robot == 'leaphand': # refine leaphand's penetration
                                deep = torch.nonzero(penetration_after > 0.005).flatten()
                                if deep.numel():
                                    candidate, _, candidate_depth = refine_grasps(
                                        refined[deep], object_name, self.asset_dir, robot)
                                    better = candidate_depth < penetration_after[deep]
                                    refined[deep[better]] = candidate[better]
                                    penetration_after[deep[better]] = candidate_depth[better]
                        poses = refined.cpu()
                    if not torch.isfinite(poses).all():
                        raise RuntimeError(f'Non-finite samples: {robot}/{object_name}')
                    records = [{'q': q, 'object_name': object_name, 'robot_name': robot,
                                'masked_fingers': self.masked_fingers} for q in poses]
                    if penetration_after is not None:
                        for record, before, after in zip(records, penetration_before, penetration_after):
                            record['penetration_before_m'] = float(before)
                            record['penetration_after_m'] = float(after)
                    output_dir = os.path.join(save_dir, robot)
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(records, os.path.join(output_dir, f'{object_name}.pt'))
                    progress.update(len(records))


def create_visualizer(cfg):
    return VISUALIZER.get(cfg.name)(cfg)
