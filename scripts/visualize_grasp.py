#!/usr/bin/env python3
"""Write a self-contained Plotly view for one saved grasp."""

import argparse
from pathlib import Path
import sys

import plotly.graph_objects as go
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Saved grasp .pt file")
    parser.add_argument("--robot", required=True, choices=("shadowhand", "allegro", "barrett", "leaphand", "inspire"))
    parser.add_argument("--asset-dir", default="data/MultiDex_filtered")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    grasps = torch.load(args.input, map_location="cpu", weights_only=True)
    if not 0 <= args.index < len(grasps):
        parser.error(f"--index must be between 0 and {len(grasps) - 1}")
    grasp = grasps[args.index]
    if grasp['robot_name'] != args.robot:
        parser.error(f"Grasp belongs to {grasp['robot_name']}, not {args.robot}")
    q = grasp["q"].to(device=args.device, dtype=torch.float32).unsqueeze(0)
    hand_model = get_handmodel(1, args.device, args.robot, 1.0, None)
    import trimesh
    dataset, name = grasp["object_name"].split('+', 1)
    object_mesh = trimesh.load(Path(args.asset_dir) / 'object' / dataset / name / f'{name}.stl', force='mesh', process=False)

    traces = [plot_mesh(object_mesh, color="#8ecae6", opacity=0.8)]
    traces.extend(hand_model.get_plotly_data(q=q, color="#e76f51", opacity=0.9))
    figure = go.Figure(traces)
    figure.update_layout(
        title=f"{args.robot} | {grasp['object_name']} | sample {args.index}",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True)
    print(f"Visualization saved to {output}")


if __name__ == "__main__":
    main()
