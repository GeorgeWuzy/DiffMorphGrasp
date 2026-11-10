#!/usr/bin/env python3
"""Evaluate success rate and successful-grasp diversity in simulation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Grasp .pt file or single-hand directory")
    parser.add_argument("--robot", required=True, choices=("shadowhand", "allegro", "barrett", "leaphand", "inspire"), help="Hand to evaluate")
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "data/MultiDex_filtered", help="Processed dataset directory")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--use-gui", action="store_true", help="Display the simulation viewer")
    return parser.parse_args()


def evaluate_file(path, robot, asset_dir, gpu=0, use_gui=False):
    # Isaac Gym must load before PyTorch.
    from eval.simulation.validator import IsaacValidator
    from eval.metrics import pose_diversity
    import torch

    grasps = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(grasps, list) or not grasps:
        raise ValueError(f"No grasp records in {path}")
    objects = {item["object_name"] for item in grasps}
    robots = {item["robot_name"] for item in grasps}
    if len(objects) != 1 or robots != {robot}:
        raise ValueError(f"Mixed or mismatched grasp metadata in {path}: {objects}, {robots}")
    object_name = next(iter(objects))
    dataset, name = object_name.split("+", 1)
    object_urdf = Path(asset_dir) / "object" / dataset / name / f"{name}.urdf"
    meta = json.loads((ROOT / "data/urdf/urdf_assets_meta_extended.json").read_text())
    robot_urdf = ROOT / meta["urdf_path"][robot]
    if not object_urdf.is_file() or not robot_urdf.is_file():
        raise FileNotFoundError(f"Missing asset: {object_urdf} or {robot_urdf}")
    q = torch.stack([item["q"] for item in grasps]).float()
    simulator = IsaacValidator(robot, robot_urdf, len(grasps), gpu, use_gui)
    try:
        simulator.set_asset(object_urdf)
        simulator.create_envs()
        simulator.set_actor_pose_dof(q)
        success, q_grasped = simulator.run_sim()
    finally:
        simulator.destroy()
    count = int(success.sum())
    return {
        "object_name": object_name,
        "robot_name": robot,
        "num_grasps": len(grasps),
        "success_count": count,
        "success_rate": count / len(grasps),
        **pose_diversity(q_grasped, success),
    }


def format_diversity(value):
    return "N/A" if value is None else f"{value:.6f}"


def main():
    args = parse_args()
    files = [args.input] if args.input.is_file() else sorted(args.input.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files under {args.input}")
    results = []
    for path in files:
        result = evaluate_file(path, args.robot, args.asset_dir, args.gpu, args.use_gui)
        results.append(result)
        count, num_grasps = result['success_count'], result['num_grasps']
        print(
            f"[{args.robot}] {result['object_name']} | "
            f"success={count}/{num_grasps} ({count / num_grasps:.2%}) | "
            f"diversity={format_diversity(result['diversity'])}",
            flush=True,
        )
    total = sum(r["num_grasps"] for r in results)
    successes = sum(r["success_count"] for r in results)
    diversity = [r["diversity"] for r in results if r["diversity"] is not None]
    mean_diversity = sum(diversity) / len(diversity) if diversity else None
    print(
        f"[{args.robot}] overall | success={successes}/{total} ({successes / total:.2%}) | "
        f"diversity={format_diversity(mean_diversity)} "
        f"(valid objects: {len(diversity)}/{len(results)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
