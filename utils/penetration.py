from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
import trimesh

from utils.handmodel import get_handmodel
from utils.retarget import couple_inspire_joints


def should_optimize_penetration(robot, setting):
    if setting is None:
        return robot in ("allegro", "barrett", "leaphand")
    return bool(setting)


def object_sdf(asset_dir, object_name, device, pitch=0.001, padding=8):
    dataset, name = object_name.split("+", 1)
    mesh_path = Path(asset_dir) / "object" / dataset / name / f"{name}.stl"
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    voxels = mesh.voxelized(pitch=pitch).fill()
    occupied = np.pad(voxels.matrix, padding, constant_values=False)
    outside = distance_transform_edt(~occupied).astype(np.float32) * pitch
    inside = distance_transform_edt(occupied).astype(np.float32) * pitch
    sdf = outside - inside
    origin = voxels.transform[:3, 3] - padding * pitch
    field = torch.from_numpy(np.ascontiguousarray(sdf.transpose(2, 1, 0)))[None, None].to(device)
    return field, torch.as_tensor(origin, device=device, dtype=torch.float32), pitch


def sample_sdf(points, field, origin, pitch):
    shape = torch.tensor(field.shape[-1:-4:-1], device=points.device, dtype=points.dtype)
    coordinates = (points - origin) / pitch
    coordinates = 2 * coordinates / (shape - 1) - 1
    grid = coordinates.reshape(1, 1, 1, -1, 3)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True).reshape(points.shape[:-1])


def set_dense_surface_points(hand, points_per_link=512):
    batch_size = hand.batch_size
    device = hand.device
    rng = np.random.default_rng(0)
    for link_name in hand.mesh_verts:
        vertices = hand.mesh_verts[link_name]
        faces = hand.mesh_faces[link_name]
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
        choices = rng.choice(len(faces), size=points_per_link, p=areas / areas.sum())
        selected = triangles[choices]
        u = np.sqrt(rng.random((points_per_link, 1)))
        v = rng.random((points_per_link, 1))
        samples = (1 - u) * selected[:, 0] + u * (1 - v) * selected[:, 1] + u * v * selected[:, 2]
        homogeneous = np.column_stack((samples, np.ones(points_per_link)))
        hand.surface_points[link_name] = torch.as_tensor(homogeneous, device=device, dtype=torch.float32)[None].expand(batch_size, -1, -1)
    return hand


def dense_hand_model(batch_size, robot, device, points_per_link=512):
    hand = get_handmodel(batch_size, device, robot, 1.0, None, use_removed_links=False)
    return set_dense_surface_points(hand, points_per_link)


def refine_grasps(q, object_name, asset_dir, robot, steps=150, pitch=0.001):
    device = q.device
    field, origin, pitch = object_sdf(asset_dir, object_name, device, pitch)
    hand = dense_hand_model(len(q), robot, device)
    initial = q.detach()
    translation = initial[:, :3].clone().requires_grad_(True)
    rotation = initial[:, 3:9].clone().requires_grad_(True)
    joints = initial[:, 9:].clone().requires_grad_(True)
    lower = hand.revolute_joints_q_lower
    upper = hand.revolute_joints_q_upper
    optimizer = torch.optim.Adam([
        {"params": translation, "lr": 0.0015},
        {"params": rotation, "lr": 0.004},
        {"params": joints, "lr": 0.006},
    ])
    if robot == "inspire":
        steps = min(steps, 25)
    for step in range(steps):
        optimizer.zero_grad()
        pose_joints = couple_inspire_joints(joints) if robot == "inspire" else joints
        pose = torch.cat((translation, rotation, pose_joints), dim=-1)
        points = hand.get_surface_points(pose)
        distances = sample_sdf(points, field, origin, pitch)
        penetration = (0.001 - distances).clamp_min(0)
        worst = penetration.square().topk(min(32, penetration.shape[1]), dim=1).values.mean()
        position_change = (translation - initial[:, :3]).square().sum(dim=1).mean()
        rotation_change = (rotation - initial[:, 3:9]).square().sum(dim=1).mean()
        joint_change = (joints - initial[:, 9:]).square().sum(dim=1).mean()
        loss = worst
        if step < steps - 50:
            loss = loss + 0.001 * position_change + 0.00001 * rotation_change + 0.00001 * joint_change
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            joints.clamp_(lower, upper)
            if robot == "inspire":
                translation.clamp_(initial[:, :3] - 0.002, initial[:, :3] + 0.002)
                rotation.clamp_(initial[:, 3:9] - 0.03, initial[:, 3:9] + 0.03)
                joints.clamp_(initial[:, 9:] - 0.03, initial[:, 9:] + 0.03)
    with torch.no_grad():
        before = sample_sdf(hand.get_surface_points(initial), field, origin, pitch)
        pose_joints = couple_inspire_joints(joints) if robot == "inspire" else joints
        refined = torch.cat((translation, rotation, pose_joints), dim=-1).detach()
        after = sample_sdf(hand.get_surface_points(refined), field, origin, pitch)
    return refined, (-before.min(dim=1).values).clamp_min(0), (-after.min(dim=1).values).clamp_min(0)
