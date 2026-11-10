import torch

from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d, rot_to_orthod6d


def align_leap_root(q: torch.Tensor) -> torch.Tensor:
    rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
    canonical = q.new_tensor([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    aligned = q.clone()
    aligned[:, 3:9] = rot_to_orthod6d(rotation @ canonical)
    aligned[:, :3] += rotation @ q.new_tensor([-0.024, 0.037, 0.002])
    return aligned


def align_inspire_root(q: torch.Tensor) -> torch.Tensor:
    rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
    canonical = q.new_tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    aligned = q.clone()
    aligned[:, 3:9] = rot_to_orthod6d(rotation @ canonical)
    aligned[:, :3] += rotation @ q.new_tensor([0., -0.01, 0.232])
    return aligned


def couple_inspire_joints(joints: torch.Tensor) -> torch.Tensor:
    coupled = joints.clone()
    coupled[..., 2] = 1.334 * joints[..., 1]
    coupled[..., 3] = 0.667 * joints[..., 1]
    for driver in (4, 6, 8, 10):
        coupled[..., driver + 1] = 1.06399 * joints[..., driver] - 0.04545
    return coupled
