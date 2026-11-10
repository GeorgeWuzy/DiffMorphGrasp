"""Pose conversions used by the simulation evaluator."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def to_euler(q, dof):
    if q.ndim != 2 or not torch.isfinite(q).all():
        raise ValueError("Expected finite batched grasp poses")
    if q.shape[1] == dof:
        return q.float()
    if q.shape[1] != dof + 3:
        raise ValueError(f"Expected {dof} Euler or {dof + 3} rot6d values, got {q.shape[1]}")
    rotation = q[:, 3:9].double()
    if (rotation[:, :3].norm(dim=-1) < 1e-12).any() or (rotation[:, 3:].norm(dim=-1) < 1e-12).any():
        raise ValueError("Degenerate 6D rotation")
    x = torch.nn.functional.normalize(rotation[:, :3], dim=-1)
    y = torch.nn.functional.normalize(rotation[:, 3:], dim=-1)
    if ((x + y).norm(dim=-1) < 1e-8).any() or ((x - y).norm(dim=-1) < 1e-8).any():
        raise ValueError("Degenerate 6D rotation")
    a = torch.nn.functional.normalize(x + y, dim=-1)
    b = torch.nn.functional.normalize(x - y, dim=-1)
    x = torch.nn.functional.normalize(a + b, dim=-1)
    y = torch.nn.functional.normalize(a - b, dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, y, dim=-1), dim=-1)
    matrix = torch.stack((x, y, z), dim=-1).cpu().numpy()
    euler = torch.as_tensor(Rotation.from_matrix(matrix).as_euler("XYZ"), dtype=q.dtype, device=q.device)
    return torch.cat((q[:, :3], euler, q[:, 9:]), dim=-1).float()


def relative_pose(object_pose, wrist):
    object_pose = object_pose.cpu().numpy()
    wrist = wrist.cpu().numpy().copy()
    valid = np.isfinite(object_pose).all(axis=1) & (np.linalg.norm(object_pose[:, 3:7], axis=1) > 1e-12)
    result = np.full_like(wrist, np.nan)
    if valid.any():
        inverse = Rotation.from_quat(object_pose[valid, 3:7]).inv()
        result[valid, :3] = inverse.apply(wrist[valid, :3] - object_pose[valid, :3])
        result[valid, 3:6] = (inverse * Rotation.from_euler("XYZ", wrist[valid, 3:6])).as_euler("XYZ")
        result[valid, 6:] = wrist[valid, 6:]
    return torch.from_numpy(result)
