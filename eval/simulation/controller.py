"""Grasp controller, adapted from code by GeorgeWuzy (MIT)."""

import torch

from utils.retarget import couple_inspire_joints


def link_direction(robot, joint):
    if joint.startswith("virtual") or joint == "":
        return None
    if robot == "barrett":
        return None if joint in ("bh_j11_joint", "bh_j21_joint") else (-1., 0., 0.)
    if robot == "allegro":
        return None if joint in ("joint_0.0", "joint_4.0", "joint_8.0", "joint_13.0") else (0., 0., 1.)
    if robot == "shadowhand":
        if joint in ("WRJ2", "WRJ1"):
            return None
        return (1., 0., 0.) if joint == "THJ5" else (0., 0., 1.)
    if robot == "inspire":
        if joint == "thumb_proximal_yaw_joint":
            return None
        return (1., 0., 0.) if joint.startswith("thumb_") else (0., 1., 0.)
    if robot == "leaphand":
        if joint in ("0", "4", "8", "13"):
            return None
        return (0., 1., 0.) if joint in ("1", "5", "9", "12", "14") else (0., -1., 0.)
    raise ValueError(robot)


def controller(robot, q, chain):
    names = list(chain.get_joint_parameter_names())
    if q.ndim != 2 or q.shape[1] != len(names):
        raise ValueError("Controller expects root XYZ, intrinsic XYZ Euler angles, and hand joints")
    status = chain.forward_kinematics(q)
    lower, upper = [torch.as_tensor(v, dtype=q.dtype, device=q.device) for v in chain.get_joint_limits()]
    outer, inner = q.clone(), q.clone()
    outer_scale, inner_scale = (0., 0.05) if robot == "barrett" else (0.1, 0.2)
    for frame_name in chain.get_frame_names():
        frame = chain.find_frame(frame_name)
        if frame.joint.name not in names:
            continue
        direction = link_direction(robot, frame.joint.name)
        if direction is None:
            continue
        transform = status[frame_name].get_matrix()
        axis = transform[:, :3, :3] @ frame.joint.axis.to(q)
        tangent = transform[:, :3, :3] @ q.new_tensor(direction)
        normal = torch.cross(axis, tangent, dim=-1)
        # The object is at the origin; extended FK includes the wrist pose.
        origin = transform[:, :3, 3]
        toward_object = -origin / origin.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        close_upper = (normal * toward_object).sum(dim=-1) >= 0
        index = names.index(frame.joint.name)
        open_limit = torch.where(close_upper, lower[index], upper[index])
        close_limit = torch.where(close_upper, upper[index], lower[index])
        outer[:, index] += outer_scale * (open_limit - q[:, index])
        inner[:, index] += inner_scale * (close_limit - q[:, index])
    if robot == "inspire":
        outer[:, 6:] = couple_inspire_joints(outer[:, 6:])
        inner[:, 6:] = couple_inspire_joints(inner[:, 6:])
    return outer, inner
