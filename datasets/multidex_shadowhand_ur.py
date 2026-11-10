from typing import Any, Tuple, Dict
import os
import json
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
from utils.myutils import *
from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
from datasets.transforms import make_default_transform
from datasets.base import DATASET
from termcolor import cprint
from utils.robots import validate_hand_name
import pytorch_kinematics as pk
@DATASET.register()
class MultiDexShadowHandUR(Dataset):

    _train_split = [
                    "contactdb+alarm_clock", "contactdb+banana", "contactdb+binoculars",
                    "contactdb+cell_phone",
                    "contactdb+cube_large",
                    "contactdb+cube_medium",
                    "contactdb+cube_small",
                    "contactdb+cylinder_large",
                    "contactdb+cylinder_small",
                    "contactdb+elephant", "contactdb+flashlight", "contactdb+hammer",
                    "contactdb+light_bulb", "contactdb+mouse", "contactdb+piggy_bank", "contactdb+ps_controller",
                    "contactdb+pyramid_large",
                    "contactdb+pyramid_medium",
                    "contactdb+pyramid_small",
                    "contactdb+stanford_bunny", "contactdb+stapler", "contactdb+toothpaste", "contactdb+torus_large",
                    "contactdb+torus_medium",
                    "contactdb+torus_small",
                    "contactdb+train",
                    "ycb+bleach_cleanser", "ycb+cracker_box", "ycb+foam_brick", "ycb+gelatin_box", "ycb+hammer",
                    "ycb+lemon", "ycb+master_chef_can", "ycb+mini_soccer_ball", "ycb+mustard_bottle", "ycb+orange",
                    "ycb+peach", "ycb+pitcher_base", "ycb+plum", "ycb+power_drill", "ycb+pudding_box",
                    "ycb+rubiks_cube", "ycb+sponge", "ycb+strawberry", "ycb+sugar_box",
                    "ycb+toy_airplane",
                    "ycb+tuna_fish_can", "ycb+wood_block",
                    ]
    _test_split = [
                    "contactdb+apple",
                    "contactdb+camera",
                    "contactdb+cylinder_medium",
                    "contactdb+rubber_duck",
                    "contactdb+door_knob",
                    "contactdb+water_bottle",
                    "ycb+baseball",
                    "ycb+pear",
                    "ycb+potted_meat_can",
                    "ycb+tomato_soup_can",
                    ]
    _all_split = ["contactdb+alarm_clock", "contactdb+banana", "contactdb+binoculars",
                    "contactdb+cell_phone", "contactdb+cube_large", "contactdb+cube_medium",
                    "contactdb+cube_small", "contactdb+cylinder_large", "contactdb+cylinder_small",
                    "contactdb+elephant", "contactdb+flashlight", "contactdb+hammer",
                    "contactdb+light_bulb", "contactdb+mouse", "contactdb+piggy_bank", "contactdb+ps_controller",
                    "contactdb+pyramid_large", "contactdb+pyramid_medium", "contactdb+pyramid_small",
                    "contactdb+stanford_bunny", "contactdb+stapler", "contactdb+toothpaste", "contactdb+torus_large",
                    "contactdb+torus_medium", "contactdb+torus_small", "contactdb+train",
                    "ycb+bleach_cleanser", "ycb+cracker_box", "ycb+foam_brick", "ycb+gelatin_box", "ycb+hammer",
                    "ycb+lemon", "ycb+master_chef_can", "ycb+mini_soccer_ball", "ycb+mustard_bottle", "ycb+orange",
                    "ycb+peach", "ycb+pitcher_base", "ycb+plum", "ycb+power_drill", "ycb+pudding_box",
                    "ycb+rubiks_cube", "ycb+sponge", "ycb+strawberry", "ycb+sugar_box", "ycb+toy_airplane",
                    "ycb+tuna_fish_can", "ycb+wood_block", "contactdb+apple", "contactdb+camera", "contactdb+cylinder_medium", "contactdb+rubber_duck",
                    "contactdb+door_knob",  "contactdb+water_bottle", "ycb+baseball", "ycb+pear", "ycb+potted_meat_can",
                    "ycb+tomato_soup_can"]



    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(MultiDexShadowHandUR, self).__init__()

        overlap = set(self._train_split).intersection(self._test_split)
        if overlap:
            raise ValueError(f"Train/test object splits overlap: {sorted(overlap)}")

        assert isinstance(cfg.robot_name, str), f"Dataset expects single robot_name, got {type(cfg.robot_name)}"

        self.robot_name = cfg.robot_name
        validate_hand_name(self.robot_name)
        self.robot_class_map = {
                    'allegro': 0,
                    'barrett': 1,
                    'shadowhand': 2
                }

        self.phase = phase
        self.slurm = slurm
        if self.phase == 'train':
            self.split = self._train_split
        elif self.phase == 'test':
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.datasetname = cfg.name
        self.device = cfg.device
        self.is_downsample = cfg.is_downsample
        self.modeling_keys = cfg.modeling_keys
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.use_normal = cfg.use_normal
        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans
        self.masked_fingers = cfg.masked_fingers
        self.obj_dim = int(3 + 3 * self.use_color + 3 * self.use_normal)
        self.transform = make_default_transform(cfg, phase)
        self.use_llm=cfg.use_llm
        self.asset_dir = cfg.asset_dir_slurm if self.slurm else cfg.asset_dir
        self.data_dir = os.path.join(self.asset_dir, self.robot_name)
        self.scene_path = os.path.join(self.asset_dir, 'object_pcds_nors.pkl')

        if self.robot_name == "shadowhand":
            self._global_trans_lower = torch.tensor([-0.40145787596702576, -0.2814755141735077, -0.7249335646629333]).cpu()
            self._global_trans_upper = torch.tensor([0.3077795207500458, 0.2635266184806824, -0.06277382373809814]).cpu()
            self._joint_angle_lower = torch.tensor([-0.5372000336647034, -0.7978581786155701, -0.5607982277870178, -0.021734129637479782, -0.07955564558506012, -0.04349619895219803, -0.46266934275627136, -0.04343005269765854, -0.05746321380138397, -0.03191731125116348, -0.48232829570770264, -0.046414993703365326, -0.03352443873882294, -0.05390690639615059, -0.06668327748775482, -0.5360128283500671, -0.06783388555049896, -0.10130200535058975, -0.03972449153661728, -1.0566232204437256, -0.01583007350564003, -0.27672284841537476, -0.5415423512458801, -0.07801459729671478]).cpu()
            self._joint_angle_upper = torch.tensor([0.2010698914527893, 0.6973757147789001, 0.47785425186157227, 1.592340350151062, 1.6030197143554688, 1.6148632764816284, 0.5483921766281128, 1.58761465549469, 1.6018972396850586, 1.594163417816162, 0.45345184206962585, 1.6089757680892944, 1.5927749872207642, 1.5898855924606323, 0.767267107963562, 0.501705527305603, 1.5881710052490234, 1.5814329385757446, 1.5826213359832764, 1.0589795112609863, 1.3960016965866089, 0.33850598335266113, 0.556107759475708, 1.6415008306503296]).cpu()
        elif self.robot_name == "allegro":
            self._global_trans_lower = torch.tensor([-0.1976383924484253, -0.16971074044704437, -0.1768803745508194]).cpu()
            self._global_trans_upper = torch.tensor([0.1485014408826828, 0.20096272230148315, 0.18164443969726562]).cpu()
            self._joint_angle_lower = torch.tensor([-0.4753612279891968, -0.20232343673706055, -0.17653462290763855, -0.23568208515644073, -0.48612773418426514, -0.20396627485752106, -0.1820547878742218, -0.2523525357246399, -0.49549704790115356, -0.20113831758499146, -0.18465019762516022, -0.24092593789100647, 0.2555975317955017, -0.11633563786745071, -0.1944560408592224, -0.1750565469264984]).cpu()
            self._joint_angle_upper = torch.tensor([0.48461243510246277, 1.64198899269104, 1.7253249883651733, 1.6197439432144165, 0.48350149393081665, 1.6179964542388916, 1.7139320373535156, 1.624678134918213, 0.4795088469982147, 1.6191741228103638, 1.7120823860168457, 1.6231164932250977, 1.4589459896087646, 1.184441089630127, 1.6470000743865967, 1.731457233428955]).cpu()
        elif self.robot_name == "barrett":
            self._global_trans_lower = torch.tensor([-0.17365334928035736, -0.22851286828517914, -0.3047621548175812]).cpu()
            self._global_trans_upper = torch.tensor([0.18145515024662018, 0.18716278672218323, 0.03202369809150696]).cpu()
            self._joint_angle_lower = torch.tensor([-0.0038286177441477776, -0.03031402826309204, -0.017690816894173622, -0.040012527257204056, -0.03475988656282425, -0.11426961421966553, -0.002487634774297476, -0.12228795140981674]).cpu()
            self._joint_angle_upper = torch.tensor([2.444960594177246, 1.1876929998397827, 3.147313117980957, 2.443398952484131, 0.9103814363479614, 3.1647799015045166, 2.4438071250915527, 0.9041003584861755]).cpu()
        elif self.robot_name == "inspire":
            self._global_trans_lower = torch.tensor([-0.40145787596702576, -0.2814755141735077, -0.7249335646629333])
            self._global_trans_upper = torch.tensor([0.3077795207500458, 0.2635266184806824, -0.06277382373809814])
            with open("data/urdf/inspire_hand/inspire_hand_right.urdf", "rb") as file:
                lower, upper = pk.build_chain_from_urdf(file.read()).get_joint_limits()
            self._joint_angle_lower = torch.as_tensor(lower, dtype=torch.float32)
            self._joint_angle_upper = torch.as_tensor(upper, dtype=torch.float32)
        elif self.robot_name == "leaphand":
            self._global_trans_lower = torch.tensor([-0.1976383924484253, -0.16971074044704437, -0.1768803745508194]).cpu()
            self._global_trans_upper = torch.tensor([0.1485014408826828, 0.20096272230148315, 0.18164443969726562]).cpu()
            self._joint_angle_lower = torch.tensor([-0.20232343673706055, -0.4753612279891968, -0.17653462290763855, -0.23568208515644073, -0.20396627485752106, -0.48612773418426514, -0.1820547878742218, -0.2523525357246399, -0.20113831758499146, -0.49549704790115356, -0.18465019762516022, -0.24092593789100647, 0.2555975317955017, -0.11633563786745071, -0.1944560408592224, -0.1750565469264984]).cpu()
            self._joint_angle_upper = torch.tensor([1.64198899269104, 0.48461243510246277, 1.7253249883651733, 1.6197439432144165, 1.6179964542388916, 0.48350149393081665, 1.7139320373535156, 1.624678134918213, 1.6191741228103638, 0.4795088469982147, 1.7120823860168457, 1.6231164932250977, 1.4589459896087646, 1.184441089630127, 1.6470000743865967, 1.731457233428955]).cpu()
            with open("data/urdf/leap_hand/robot.urdf", "rb") as file:
                lower, upper = pk.build_chain_from_urdf(file.read()).get_joint_limits()
            self._joint_angle_lower = torch.as_tensor(lower, dtype=torch.float32)
            self._joint_angle_upper = torch.as_tensor(upper, dtype=torch.float32)
        elif self.robot_name == "allegro_leap":
            self._global_trans_lower = torch.tensor([-0.1976383924484253, -0.16971074044704437, -0.1768803745508194]).cpu()
            self._global_trans_upper = torch.tensor([0.1485014408826828, 0.20096272230148315, 0.18164443969726562]).cpu()
            self._joint_angle_lower = torch.tensor([-0.4753612279891968, -0.20232343673706055, -0.17653462290763855, -0.23568208515644073, -0.48612773418426514, -0.20396627485752106, -0.1820547878742218, -0.2523525357246399, -0.49549704790115356, -0.20113831758499146, -0.18465019762516022, -0.24092593789100647, 0.2555975317955017, -0.11633563786745071, -0.1944560408592224, -0.1750565469264984]).cpu()
            self._joint_angle_upper = torch.tensor([0.48461243510246277, 1.64198899269104, 1.7253249883651733, 1.6197439432144165, 0.48350149393081665, 1.6179964542388916, 1.7139320373535156, 1.624678134918213, 0.4795088469982147, 1.6191741228103638, 1.7120823860168457, 1.6231164932250977, 1.4589459896087646, 1.184441089630127, 1.6470000743865967, 1.731457233428955]).cpu()

        elif self.robot_name == "inspire_shadow":
            self._global_trans_lower = torch.tensor([-0.40145787596702576, -0.2814755141735077, -0.7249335646629333]).cpu()
            self._global_trans_upper = torch.tensor([0.3077795207500458, 0.2635266184806824, -0.06277382373809814]).cpu()
            self._joint_angle_lower = torch.tensor([-0.5372000336647034, -0.7978581786155701, -0.5607982277870178, -0.021734129637479782, -0.07955564558506012, -0.04349619895219803, -0.46266934275627136, -0.04343005269765854, -0.05746321380138397, -0.03191731125116348, -0.48232829570770264, -0.046414993703365326, -0.03352443873882294, -0.05390690639615059, -0.06668327748775482, -0.5360128283500671, -0.06783388555049896, -0.10130200535058975, -0.03972449153661728, -1.0566232204437256, -0.01583007350564003, -0.27672284841537476, -0.5415423512458801, -0.07801459729671478]).cpu()
            self._joint_angle_upper = torch.tensor([0.2010698914527893, 0.6973757147789001, 0.47785425186157227, 1.592340350151062, 1.6030197143554688, 1.6148632764816284, 0.5483921766281128, 1.58761465549469, 1.6018972396850586, 1.594163417816162, 0.45345184206962585, 1.6089757680892944, 1.5927749872207642, 1.5898855924606323, 0.767267107963562, 0.501705527305603, 1.5881710052490234, 1.5814329385757446, 1.5826213359832764, 1.0589795112609863, 1.3960016965866089, 0.33850598335266113, 0.556107759475708, 1.6415008306503296]).cpu()

        self.adjacency_matrix, _, self.parent_matrix, _, self.child_matrix, _ = get_matrix(self.robot_name, 'cpu', self.masked_fingers)
        self.limits, self.origin, self.axis, self.box = remap_joint_info(self.robot_name, self.masked_fingers)
        self._pre_load_data(case_only)

    def _pre_load_data(self, case_only: bool) -> None:
        self.frames = []
        self.scene_pcds = {}
        if case_only and self.robot_name in ("leaphand", "inspire"):
            self.scene_pcds = pickle.load(open(self.scene_path, 'rb'))
            self.frames = [{'object_name': name} for name in self._test_split if name in self.scene_pcds]
            return
        grasp_dataset = torch.load(os.path.join(self.data_dir, f'{self.robot_name}_new.pt'))
        self.scene_pcds = pickle.load(open(self.scene_path, 'rb'))

        if self.use_llm:
            scene_text_file = os.path.join(self.asset_dir, "multidex_gpt4o_mini.json")
            with open(scene_text_file, "r") as jsonfile:
                self.scene_text = json.load(jsonfile)
            for k, text in self.scene_text.items():
                txtclips = text.split("\n")
                self.scene_text[k] = txtclips[:]


        self.dataset_info = grasp_dataset['info']
        for obj in grasp_dataset['info']['num_per_object'].keys():
            if obj not in self.split:
                self.dataset_info['num_per_object'][obj] = 0
        for mdata in grasp_dataset['metadata']:
            mdata_qpos = mdata[0].cpu()
            joint_angle = mdata_qpos.clone().detach()[9:]
            global_trans = mdata_qpos.clone().detach()[:3]
            if self.normalize_x:
                joint_angle, joint_padding_mask = self.angle_normalize(joint_angle)
            if self.normalize_x_trans:
                global_trans = self.trans_normalize(global_trans)
            mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)

            global_trans_mask = torch.ones(3, dtype=torch.float32)
            padding_mask = torch.cat([global_trans_mask, joint_padding_mask], dim=0)

            if mdata[2] in self.split:
                self.frames.append({'robot_name': self.robot_name,
                                    'object_name': mdata[2],
                                    'object_rot_mat': mdata[1].clone().detach().numpy(),
                                    'qpos': mdata_qpos,
                                    'padding_mask': padding_mask})

    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm

    def angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self._joint_angle_lower), (self._joint_angle_upper - self._joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2

        map_name = 'shadowhand' if self.robot_name == 'inspire_shadow' else self.robot_name
        unified_joint_norm, padding_mask = joint_to_unified_dim(joint_angle_norm, map_name, self.masked_fingers)

        return unified_joint_norm, padding_mask

    def angle_denormalize(self, unified_joint_norm: torch.Tensor):
        if self.robot_name == 'inspire_shadow':
            _, active_mask = joint_to_unified_dim(
                torch.ones_like(unified_joint_norm), 'shadowhand', self.masked_fingers)
            joint_angle_denorm = unified_joint_norm + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
            joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
            joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
            joint_angle_denorm = joint_angle_denorm.masked_fill(active_mask == 0, 0.)
            joint_angle_denorm = unified_to_joint_dim(joint_angle_denorm, self.robot_name, self.masked_fingers)
        else:
            joint_angle_norm = unified_to_joint_dim(unified_joint_norm, self.robot_name, self.masked_fingers)
            active_mask = unified_to_joint_dim(
                torch.ones_like(unified_joint_norm), self.robot_name, self.masked_fingers)
            joint_angle_denorm = joint_angle_norm + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
            joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
            joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
            joint_angle_denorm = joint_angle_denorm.masked_fill(active_mask == 0, 0.)

        return joint_angle_denorm

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: Any) -> Tuple:

        frame = self.frames[index]

        scene_id = frame['object_name']
        scene_rot_mat = frame['object_rot_mat']
        scene_pc = self.scene_pcds[scene_id]
        nor = np.einsum('mn, kn->km', scene_rot_mat, scene_pc[:,3:6])
        scene_pc = np.einsum('mn, kn->km', scene_rot_mat, scene_pc[:,:3])
        cam_tran = None

        if self.phase != 'train':
            np.random.seed(0)
        resample_indices = np.random.permutation(len(scene_pc))
        scene_pc = scene_pc[resample_indices[:self.num_points]]
        nor = nor[resample_indices[:self.num_points]]
        xyz = scene_pc[:, 0:3]
        nor = nor[:, 0:3]
        if self.use_color:
            color = scene_pc[:, 3:6] / 255.
            feat = np.concatenate([color], axis=-1)

        grasp_qpos = (
            frame['qpos']
        )
        padding_mask = frame['padding_mask']

        data = {
            'x': grasp_qpos,
            'padding_mask': padding_mask,
            'pos': xyz,
            'scene_rot_mat': scene_rot_mat,
            'cam_tran': cam_tran,
            'scene_id': scene_id,
            'normal': nor,
            'robot_name': self.robot_name,
            'adjacency_matrix': self.adjacency_matrix,
            'parent_matrix': self.parent_matrix,
            'child_matrix': self.child_matrix,
            'joint_limits': self.limits,
            'joint_origin': self.origin,
            'joint_axis': self.axis,
            'joint_box': self.box,
            'masked_fingers': self.masked_fingers
        }
        if self.use_normal:
            normal = nor
            feat = np.concatenate([normal], axis=-1)
            data['feat'] = feat
        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys)

        if self.use_llm:
            data['text'] = self.scene_text[scene_id]

        data = detach_tensors(data)
        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)

def detach_tensors(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_tensors(v) for v in obj]
    return obj

