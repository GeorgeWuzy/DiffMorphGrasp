import torch


def pose_diversity(q, success):
    selected = q[success]
    if len(selected) < 2:
        return {"diversity": None, "joint_diversity_rad": None, "num_successful": len(selected)}
    if not torch.isfinite(selected).all():
        raise ValueError("Non-finite successful poses")
    std = selected.double().std(dim=0, correction=1)
    return {
        "diversity": std.mean().item(),
        "joint_diversity_rad": std[6:].mean().item(),
        "num_successful": len(selected),
    }
