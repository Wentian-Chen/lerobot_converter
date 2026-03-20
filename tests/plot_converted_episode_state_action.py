#!/usr/bin/env python3
"""Plot state/action trajectories for one converted episode stored as step-wise HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _load_episode_arrays(dataset_root: Path, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    episode_dir = dataset_root / "episodes" / f"{episode_id:06d}" / "steps"
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode steps directory not found: {episode_dir}")

    step_dirs = sorted(p for p in episode_dir.iterdir() if p.is_dir())
    if not step_dirs:
        raise ValueError(f"No step directories found in: {episode_dir}")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for step_dir in step_dirs:
        h5_path = step_dir / "data.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing step file: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            if "observation" not in f or "proprio" not in f["observation"]:
                raise KeyError(f"Missing observation/proprio in {h5_path}")
            if "action" not in f:
                raise KeyError(f"Missing action in {h5_path}")

            state = np.asarray(f["observation/proprio"], dtype=np.float64).reshape(-1)
            action = np.asarray(f["action"], dtype=np.float64).reshape(-1)

        if state.shape[0] != 7:
            raise ValueError(f"state dim is {state.shape[0]} in {h5_path}, expected 7")
        if action.shape[0] != 7:
            raise ValueError(f"action dim is {action.shape[0]} in {h5_path}, expected 7")

        states.append(state)
        actions.append(action)

    return np.vstack(states), np.vstack(actions)


def _plot_trajectory(data: np.ndarray, title: str, out_path: Path, joint_names: list[str]) -> None:
    x = np.arange(data.shape[0])
    fig, axes = plt.subplots(data.shape[1], 1, figsize=(14, 2.1 * data.shape[1]), sharex=True)
    if data.shape[1] == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x, data[:, i], linewidth=1.0)
        ax.set_ylabel(joint_names[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("step_index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one converted episode's state/action curves.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/home/lxx/repo/datasets/dream-adapter/miku112/pick_banana_200_newTable_Binary_converted"),
        help="Converted dataset root path.",
    )
    parser.add_argument("--episode_id", type=int, default=0, help="Episode id, e.g. 0 for 000000")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/lxx/repo/lerobot_converter/tests"),
        help="Directory to save output figures.",
    )
    args = parser.parse_args()

    states, actions = _load_episode_arrays(args.dataset_root, args.episode_id)

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
    state_out = args.output_dir / f"converted_episode_{args.episode_id:06d}_state.png"
    action_out = args.output_dir / f"converted_episode_{args.episode_id:06d}_action.png"

    _plot_trajectory(
        data=states,
        title=f"Episode {args.episode_id:06d} - observation/proprio",
        out_path=state_out,
        joint_names=joint_names,
    )
    _plot_trajectory(
        data=actions,
        title=f"Episode {args.episode_id:06d} - action",
        out_path=action_out,
        joint_names=joint_names,
    )

    print(f"Loaded steps: {states.shape[0]}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Saved state plot: {state_out}")
    print(f"Saved action plot: {action_out}")


if __name__ == "__main__":
    main()
