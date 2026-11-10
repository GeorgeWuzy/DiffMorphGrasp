from typing import Dict
import os
import numpy as np
import torch
from omegaconf import DictConfig
from models.optimizer.optimizer import Optimizer
from models.base import OPTIMIZER
import pickle
import pytorch_kinematics as pk
import torch.functional as F
from utils.handmodel import get_handmodel, ERF_loss, SPF_loss, SRF_loss
from einops import rearrange
from termcolor import cprint
from utils.myutils import unified_to_joint_dim, joint_to_unified_dim
from utils.penetration import dense_hand_model, object_sdf, sample_sdf, set_dense_surface_points, should_optimize_penetration
from utils.retarget import align_inspire_root, align_leap_root, couple_inspire_joints

@OPTIMIZER.register()
class GraspWithObject(Optimizer):

    _N_OBJ = 4096
    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:

        self.robot_name = cfg.robot_name

        self.robot_configs = {
            "shadowhand": {
                "global_trans_lower": torch.tensor([-0.40145787596702576, -0.2814755141735077, -0.7249335646629333]).cuda(),
                "global_trans_upper": torch.tensor([0.3077795207500458, 0.2635266184806824, -0.06277382373809814]).cuda(),
                "joint_angle_lower": torch.tensor([-0.5372000336647034, -0.7978581786155701, -0.5607982277870178, -0.021734129637479782, -0.07955564558506012, -0.04349619895219803, -0.46266934275627136, -0.04343005269765854, -0.05746321380138397, -0.03191731125116348, -0.48232829570770264, -0.046414993703365326, -0.03352443873882294, -0.05390690639615059, -0.06668327748775482, -0.5360128283500671, -0.06783388555049896, -0.10130200535058975, -0.03972449153661728, -1.0566232204437256, -0.01583007350564003, -0.27672284841537476, -0.5415423512458801, -0.07801459729671478]).cuda(),
                "joint_angle_upper": torch.tensor([0.2010698914527893, 0.6973757147789001, 0.47785425186157227, 1.592340350151062, 1.6030197143554688, 1.6148632764816284, 0.5483921766281128, 1.58761465549469, 1.6018972396850586, 1.594163417816162, 0.45345184206962585, 1.6089757680892944, 1.5927749872207642, 1.5898855924606323, 0.767267107963562, 0.501705527305603, 1.5881710052490234, 1.5814329385757446, 1.5826213359832764, 1.0589795112609863, 1.3960016965866089, 0.33850598335266113, 0.556107759475708, 1.6415008306503296]).cuda()
            },
            "allegro": {
                "global_trans_lower": torch.tensor([-0.1976383924484253, -0.16971074044704437, -0.1768803745508194]).cuda(),
                "global_trans_upper": torch.tensor([0.1485014408826828, 0.20096272230148315, 0.18164443969726562]).cuda(),
                "joint_angle_lower": torch.tensor([-0.4753612279891968, -0.20232343673706055, -0.17653462290763855, -0.23568208515644073, -0.48612773418426514, -0.20396627485752106, -0.1820547878742218, -0.2523525357246399, -0.49549704790115356, -0.20113831758499146, -0.18465019762516022, -0.24092593789100647, 0.2555975317955017, -0.11633563786745071, -0.1944560408592224, -0.1750565469264984]).cuda(),
                "joint_angle_upper": torch.tensor([0.48461243510246277, 1.64198899269104, 1.7253249883651733, 1.6197439432144165, 0.48350149393081665, 1.6179964542388916, 1.7139320373535156, 1.624678134918213, 0.4795088469982147, 1.6191741228103638, 1.7120823860168457, 1.6231164932250977, 1.4589459896087646, 1.184441089630127, 1.6470000743865967, 1.731457233428955]).cuda()
            },
            "leaphand": {
                "global_trans_lower": torch.tensor([-0.1976383924484253, -0.16971074044704437, -0.1768803745508194]).cuda(),
                "global_trans_upper": torch.tensor([0.1485014408826828, 0.20096272230148315, 0.18164443969726562]).cuda(),
                "joint_angle_lower": torch.tensor([-0.20232343673706055, -0.4753612279891968, -0.17653462290763855, -0.23568208515644073, -0.20396627485752106, -0.48612773418426514, -0.1820547878742218, -0.2523525357246399, -0.20113831758499146, -0.49549704790115356, -0.18465019762516022, -0.24092593789100647, 0.2555975317955017, -0.11633563786745071, -0.1944560408592224, -0.1750565469264984]).cuda(),
                "joint_angle_upper": torch.tensor([1.64198899269104, 0.48461243510246277, 1.7253249883651733, 1.6197439432144165, 1.6179964542388916, 0.48350149393081665, 1.7139320373535156, 1.624678134918213, 1.6191741228103638, 0.4795088469982147, 1.7120823860168457, 1.6231164932250977, 1.4589459896087646, 1.184441089630127, 1.6470000743865967, 1.731457233428955]).cuda(),
            },
            "allegro_leap": {
                "global_trans_lower": torch.tensor([-0.1976383924484253, -0.16971074044704437, -0.1768803745508194]).cuda(),
                "global_trans_upper": torch.tensor([0.1485014408826828, 0.20096272230148315, 0.18164443969726562]).cuda(),
                "joint_angle_lower": torch.tensor([-0.4753612279891968, -0.20232343673706055, -0.17653462290763855, -0.23568208515644073, -0.48612773418426514, -0.20396627485752106, -0.1820547878742218, -0.2523525357246399, -0.49549704790115356, -0.20113831758499146, -0.18465019762516022, -0.24092593789100647, 0.2555975317955017, -0.11633563786745071, -0.1944560408592224, -0.1750565469264984]).cuda(),
                "joint_angle_upper": torch.tensor([0.48461243510246277, 1.64198899269104, 1.7253249883651733, 1.6197439432144165, 0.48350149393081665, 1.6179964542388916, 1.7139320373535156, 1.624678134918213, 0.4795088469982147, 1.6191741228103638, 1.7120823860168457, 1.6231164932250977, 1.4589459896087646, 1.184441089630127, 1.6470000743865967, 1.731457233428955]).cuda(),
            },
            "barrett": {
                "global_trans_lower": torch.tensor([-0.17365334928035736, -0.22851286828517914, -0.3047621548175812]).cuda(),
                "global_trans_upper": torch.tensor([0.18145515024662018, 0.18716278672218323, 0.03202369809150696]).cuda(),
                "joint_angle_lower": torch.tensor([-0.0038286177441477776, -0.03031402826309204, -0.017690816894173622, -0.040012527257204056, -0.03475988656282425, -0.11426961421966553, -0.002487634774297476, -0.12228795140981674]).cuda(),
                "joint_angle_upper": torch.tensor([2.444960594177246, 1.1876929998397827, 3.147313117980957, 2.443398952484131, 0.9103814363479614, 3.1647799015045166, 2.4438071250915527, 0.9041003584861755]).cuda()
            },
            "inspire_shadow": {
                "global_trans_lower": torch.tensor([-0.40145787596702576, -0.2814755141735077, -0.7249335646629333]).cuda(),
                "global_trans_upper": torch.tensor([0.3077795207500458, 0.2635266184806824, -0.06277382373809814]).cuda(),
                "joint_angle_lower": torch.tensor([-0.5372000336647034, -0.7978581786155701, -0.5607982277870178, -0.021734129637479782, -0.07955564558506012, -0.04349619895219803, -0.46266934275627136, -0.04343005269765854, -0.05746321380138397, -0.03191731125116348, -0.48232829570770264, -0.046414993703365326, -0.03352443873882294, -0.05390690639615059, -0.06668327748775482, -0.5360128283500671, -0.06783388555049896, -0.10130200535058975, -0.03972449153661728, -1.0566232204437256, -0.01583007350564003, -0.27672284841537476, -0.5415423512458801, -0.07801459729671478]).cuda(),
                "joint_angle_upper": torch.tensor([0.2010698914527893, 0.6973757147789001, 0.47785425186157227, 1.592340350151062, 1.6030197143554688, 1.6148632764816284, 0.5483921766281128, 1.58761465549469, 1.6018972396850586, 1.594163417816162, 0.45345184206962585, 1.6089757680892944, 1.5927749872207642, 1.5898855924606323, 0.767267107963562, 0.501705527305603, 1.5881710052490234, 1.5814329385757446, 1.5826213359832764, 1.0589795112609863, 1.3960016965866089, 0.33850598335266113, 0.556107759475708, 1.6415008306503296]).cuda(),
            },
            "robotiq_3finger": {
                "global_trans_lower": torch.tensor([-0.1835443675518036, -0.13721005618572235, -0.3165968060493469]).cuda(),
                "global_trans_upper": torch.tensor([0.20404744148254395, 0.16013789176940918, 0.06070801988244057]).cuda(),
                "joint_angle_lower": torch.tensor([-0.00578185124322772, -1.002684473991394, -1.0411639213562012, -0.30824020504951477, -0.02103583700954914, -1.0057921409606934, -1.0085207223892212, -0.31803423166275024, -0.008431404829025269, -1.0077786445617676, -1.0078721046447754]).cuda(),
                "joint_angle_upper": torch.tensor([1.2218836545944214, 1.000183343887329, 1.0069032907485962, 0.19573935866355896, 1.2222219705581665, 1.003598928451538, 1.0241748094558716, 0.17903700470924377, 1.2226605415344238, 1.0001112222671509, 1.0111087560653687]).cuda()
            }
        }

        if "leaphand" in self.robot_name:
            with open("data/urdf/leap_hand/robot.urdf", "rb") as file:
                lower, upper = pk.build_chain_from_urdf(file.read()).get_joint_limits()
            self.robot_configs["leaphand"]["joint_angle_lower"] = torch.as_tensor(lower, dtype=torch.float32, device="cuda")
            self.robot_configs["leaphand"]["joint_angle_upper"] = torch.as_tensor(upper, dtype=torch.float32, device="cuda")
        if "inspire" in self.robot_name:
            with open("data/urdf/inspire_hand/inspire_hand_right.urdf", "rb") as file:
                lower, upper = pk.build_chain_from_urdf(file.read()).get_joint_limits()
            self.robot_configs["inspire"] = {
                "global_trans_lower": self.robot_configs["shadowhand"]["global_trans_lower"],
                "global_trans_upper": self.robot_configs["shadowhand"]["global_trans_upper"],
                "joint_angle_lower": torch.as_tensor(lower, dtype=torch.float32, device="cuda"),
                "joint_angle_upper": torch.as_tensor(upper, dtype=torch.float32, device="cuda"),
            }

        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self._BATCH_SIZE = cfg.batch_size
        self.slurm = slurm
        self.scale = cfg.grad_scale
        self.clip_grad_by_value = cfg.clip_grad_by_value
        self.modeling_keys = cfg.modeling_keys
        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans
        self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        self.guidance_scale = cfg.guidance_scale
        self.penetration_weight = cfg.penetration_weight
        self.penetration_optimization = cfg.penetration_optimization
        self._object_sdf = {}

        self.weight_ERF_loss = {
            "allegro": 0.1,
            "allegro_leap": 1.0,
            "leaphand": 1.0,
            "inspire": 0.4,
            "barrett": 0.15,
            "shadowhand": 0.4,
            "inspire_shadow": 0.4
        }
        self.weight_SPF_loss = {
            "allegro": 2.0,
            "allegro_leap": 2.0,
            "leaphand": 2.0,
            "inspire": 1.0,
            "barrett": 1.0,
            "shadowhand": 1.0,
            "inspire_shadow": 1.0
        }
        self.weight_SRF_loss = {
            "allegro": 1.0,
            "allegro_leap": 1.0,
            "leaphand": 1.0,
            "inspire": 2.0,
            "barrett": 2.0,
            "shadowhand": 2.0,
            "inspire_shadow": 2.0
        }


    def optimize(self, x: torch.Tensor, data: Dict, t: int) -> torch.Tensor:
        robot_name = data['robot_name'][0]
        obj_pcd = data['pos'].to(self.device)
        masked_fingers = data['masked_fingers'][0]
        hand_model = get_handmodel(batch_size=self._BATCH_SIZE, device=self.device, robot=robot_name, hand_scale=1., masked_fingers=masked_fingers)
        pose = x
        if robot_name in ("leaphand", "inspire"):
            identity = x.new_tensor([1., 0., 0., 0., 1., 0.]).expand(x.shape[0], -1)
            pose = torch.cat((x[:, :3], identity, x[:, 3:]), dim=-1)
            pose = align_inspire_root(pose) if robot_name == "inspire" else align_leap_root(pose)
            if robot_name == "inspire":
                pose = torch.cat((pose[:, :9], couple_inspire_joints(pose[:, 9:])), dim=-1)
        hand_pcd = hand_model.get_surface_points(q=pose).to(dtype=torch.float32)
        normal = data['normal'].clone().detach().requires_grad_(True).to(self.device)
        obj_pcd = rearrange(obj_pcd, '(b n) c -> b n c', b=self._BATCH_SIZE, n=normal.shape[1])

        obj_pcd_nor = torch.cat((obj_pcd, normal), dim=-1).to(dtype=torch.float32)
        ERF_loss_value = ERF_loss(obj_pcd_nor, hand_pcd)
        dis_keypoint = hand_model.get_dis_keypoints(q=pose)
        SPF_loss_value = SPF_loss(dis_keypoint, obj_pcd)
        hand_keypoint = hand_model.get_keypoints(q=pose)
        SRF_loss_value = SRF_loss(hand_keypoint)
        loss = self.weight_SRF_loss[robot_name] * SRF_loss_value + self.weight_SPF_loss[robot_name] * SPF_loss_value + self.weight_ERF_loss[robot_name] * ERF_loss_value
        if robot_name != "inspire" and should_optimize_penetration(robot_name, self.penetration_optimization):
            object_name = data['scene_id'][0]
            if object_name not in self._object_sdf:
                self._object_sdf[object_name] = object_sdf(self.asset_dir, object_name, self.device)
            field, origin, pitch = self._object_sdf[object_name]
            collision_hand = dense_hand_model(self._BATCH_SIZE, robot_name, self.device) if robot_name == "shadowhand" else set_dense_surface_points(hand_model)
            points = collision_hand.get_surface_points(q=pose)
            scene_rotation = data['scene_rot_mat'].to(points)
            world_points = torch.bmm(scene_rotation.transpose(1, 2), points.transpose(1, 2)).transpose(1, 2)
            distances = sample_sdf(world_points, field, origin, pitch)
            penetration = (0.001 - distances).clamp_min(0)
            loss = loss + self.penetration_weight * penetration.square().topk(64, dim=1).values.mean()
        return loss

    def gradient(self, x: torch.Tensor = None, x_0: torch.Tensor= None, data: Dict= None, x_mean: torch.Tensor= None, x_sample: torch.Tensor= None, std = None) -> torch.Tensor:
        """Guide the diffusion step using physical losses on denormalized poses."""
        assert (x_sample !=None and x != None and x_0 != None and data != None and x.shape[0] == self._BATCH_SIZE), 'x, x_0, data, x_sample must be provided and x.shape[0] == self._BATCH_SIZE'

        robot_name = data['robot_name'][0]
        masked_fingers = data['masked_fingers'][0]

        _, *x_shape= x.shape
        eps = 1e-8
        with torch.enable_grad():
            grad_list = []
            for i in range(x.shape[0] // self._BATCH_SIZE):
                i_x_0_in =x_0[i*self._BATCH_SIZE:(i+1)*self._BATCH_SIZE, :].requires_grad_(True)
                if self.normalize_x_trans:
                    i_x_0_in_denorm_trans = self.trans_denormalize(i_x_0_in[:, :3], robot_name)
                else:
                    i_x_0_in_denorm_trans = i_x_0_in[:, :3]
                if self.normalize_x:
                    i_x_0_in_denorm_angle = self.angle_denormalize(i_x_0_in[:, 3:], robot_name, masked_fingers)
                else:
                    i_x_0_in_denorm_angle = i_x_0_in[:, 3:]
                i_x_0_in_denorm = torch.cat([i_x_0_in_denorm_trans, i_x_0_in_denorm_angle], dim=-1)
                obj = self.optimize(i_x_0_in_denorm, data, t=i)
                obj = torch.linalg.norm(obj)
                i_grad = torch.autograd.grad(obj, x, retain_graph=True, allow_unused=True)[0]
                grad_list.append(i_grad)
            grad = torch.cat(grad_list, dim=0)
            grad = grad * self.scale
            gradient = torch.cat([grad[:, :3], grad[:, 3:]], dim=-1)
            grad_norm = torch.linalg.norm(gradient, dim=1).view(-1, 1)
            if std.dim() == 1:
                std = std.unsqueeze(1)
            r = torch.sqrt(torch.tensor(*x_shape)) * std[0,0]
            guidance_rate = self.guidance_scale
            d_star = -r * gradient/ (grad_norm + eps)
            if x_mean !=None:
                d_sample = x_sample -x_mean
            else:
                d_sample = std*torch.randn_like(data['x'], device=self.device)
            mix_direction = d_sample + guidance_rate * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction, dim= 1).view(-1, 1)
            mix_step = mix_direction / (mix_direction_norm + eps) * r
            if x_mean !=None:
                x_t =  x_mean + mix_step
            else:
                x_t =  x_sample + mix_step
            return x_t

    def pure_physics_optimize(self, x_norm: torch.Tensor, data: Dict, n_iters: int = 100, lr: float = 0.005):
        robot_name = data['robot_name'][0]
        masked_fingers = data['masked_fingers'][0]

        x = x_norm.detach().clone().requires_grad_(True)
        optimizer_adam = torch.optim.Adam([x], lr=lr)

        with torch.enable_grad():
            for i in range(n_iters):
                optimizer_adam.zero_grad()
                if self.normalize_x_trans:
                    trans = self.trans_denormalize(x[:, :3], robot_name)
                else:
                    trans = x[:, :3]
                if self.normalize_x:
                    angles = self.angle_denormalize(x[:, 3:], robot_name, masked_fingers)
                else:
                    angles = x[:, 3:]
                q = torch.cat([trans, angles], dim=-1)
                obj = self.optimize(q, data, t=0)
                loss = torch.linalg.norm(obj)
                loss.backward()
                optimizer_adam.step()

        return x.detach()

    def angle_denormalize(self, unified_joint_norm: torch.Tensor, robot_name: str, masked_fingers: list):
        robot_config = self.robot_configs[robot_name]
        joint_angle_lower = robot_config["joint_angle_lower"]
        joint_angle_upper = robot_config["joint_angle_upper"]

        if robot_name == 'inspire_shadow':
            joint_angle_denorm = unified_joint_norm + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
            joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
            joint_angle_denorm = joint_angle_denorm * (joint_angle_upper - joint_angle_lower) + joint_angle_lower
            joint_angle_denorm = unified_to_joint_dim(joint_angle_denorm, robot_name, masked_fingers)
        else:
            joint_angle_norm = unified_to_joint_dim(unified_joint_norm, robot_name, masked_fingers)
            joint_angle_denorm = joint_angle_norm + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
            joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
            joint_angle_denorm = joint_angle_denorm * (joint_angle_upper - joint_angle_lower) + joint_angle_lower

        return joint_angle_denorm

    def trans_denormalize(self, global_trans: torch.Tensor, robot_name: str):
        robot_config = self.robot_configs[robot_name]
        global_trans_lower = robot_config["global_trans_lower"]
        global_trans_upper = robot_config["global_trans_upper"]

        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (global_trans_upper - global_trans_lower) + global_trans_lower
        return global_trans_denorm
