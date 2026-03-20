#!/usr/bin/env python3
"""Plot per-episode state/action trajectories from a LeRobot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds


def _to_2d_array(values: Iterable[object], expected_dim: int, key: str) -> np.ndarray:
    rows = []
    for i, v in enumerate(values):
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.size != expected_dim:
            raise ValueError(
                f"{key} at row {i} has dim {arr.size}, expected {expected_dim}."
            )
        rows.append(arr)
    if not rows:
        raise ValueError(f"No values found for {key}.")
    return np.vstack(rows)


def _plot_series(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(y.shape[1], 1, figsize=(14, 2.2 * y.shape[1]), sharex=True)
    if y.shape[1] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x, y[:, i], linewidth=1.1)
        ax.set_ylabel(names[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read one episode and visualize state/action trajectories."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/lxx/repo/datasets/lerobot/miku112/pick_banana_200_newTable"),
        help="LeRobot dataset root directory.",
    )
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/lxx/repo/lerobot_converter/tests"),
        help="Directory for output figures.",
    )
    args = parser.parse_args()

    data_dir = args.dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    dataset = ds.dataset(str(data_dir), format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
        filter=(ds.field("episode_index") == args.episode_index),
    )

    if table.num_rows == 0:
        raise ValueError(f"No frames found for episode_index={args.episode_index}")

    frame_index = table["frame_index"].to_numpy()
    order = np.argsort(frame_index)
    frame_index = frame_index[order]
    timestamp = table["timestamp"].to_numpy()[order]

    state_raw = table["observation.state"].to_pylist()
    action_raw = table["action"].to_pylist()

    state = _to_2d_array(state_raw, expected_dim=7, key="observation.state")[order]
    action = _to_2d_array(action_raw, expected_dim=7, key="action")[order]

    joint_names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_out = args.output_dir / f"episode_{args.episode_index:03d}_state.png"
    action_out = args.output_dir / f"episode_{args.episode_index:03d}_action.png"

    _plot_series(
        x=frame_index,
        y=state,
        names=joint_names,
        ylabel="frame_index",
        title=f"Episode {args.episode_index} - observation.state",
        out_path=state_out,
    )
    _plot_series(
        x=frame_index,
        y=action,
        names=joint_names,
        ylabel="frame_index",
        title=f"Episode {args.episode_index} - action",
        out_path=action_out,
    )

    print(f"Frames: {len(frame_index)}")
    print(f"Timestamp range: [{float(np.min(timestamp)):.3f}, {float(np.max(timestamp)):.3f}]")
    print(f"Saved state plot: {state_out}")
    print(f"Saved action plot: {action_out}")


if __name__ == "__main__":
    main()
