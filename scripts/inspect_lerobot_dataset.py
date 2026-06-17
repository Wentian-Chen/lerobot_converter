"""Inspect one episode from a LeRobot v3.0 dataset: plot state & action trajectories jointly.

Shared dimensions (pos, quat xyz, gripper) are overlaid with both state and action lines.
State-only dimensions (quat_w, pad) are plotted alone.

Usage:
    .venv/bin/python scripts/inspect_lerobot_dataset.py \
        --dataset_root tests/results/test_libero_convert \
        --episode_index 0 \
        --output_dir tests/results/test_libero_convert/plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds


# ── Data loading ───────────────────────────────────────────────────────────


def load_episode(
    dataset_root: Path,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one episode from a LeRobot v3.0 parquet dataset.

    Returns:
        frame_index: (T,)
        state: (T, 9)  observation.state
        action: (T, 7) action
    """
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    dataset = ds.dataset(str(data_dir), format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "frame_index", "observation.state", "action"],
        filter=(ds.field("episode_index") == episode_index),
    )
    if table.num_rows == 0:
        raise ValueError(f"No frames found for episode_index={episode_index}")

    frame_index = table["frame_index"].to_numpy()
    order = np.argsort(frame_index)
    frame_index = frame_index[order]

    state = np.vstack(
        [np.asarray(v, dtype=np.float64) for v in table["observation.state"].to_pylist()]
    )[order]
    action = np.vstack(
        [np.asarray(v, dtype=np.float64) for v in table["action"].to_pylist()]
    )[order]

    return frame_index, state, action


# ── Plotting ───────────────────────────────────────────────────────────────


# Dimension groups: (subplot_title, state_indices, action_indices_or_None)
DIM_GROUPS: list[tuple[str, list[int], list[int] | None]] = [
    # Shared dims — state & action overlaid
    ("pos_x",    [0], [0]),
    ("pos_y",    [1], [1]),
    ("pos_z",    [2], [2]),
    ("quat_x",   [3], [3]),
    ("quat_y",   [4], [4]),
    ("quat_z",   [5], [5]),
    ("gripper",  [6], [6]),
    # State-only dims
    ("quat_w",   [7], None),
    ("pad",      [8], None),
]


def plot_episode(
    frame_index: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    out_path: Path,
) -> None:
    """Plot state & action trajectories jointly in a single figure."""
    n_subplots = len(DIM_GROUPS)
    fig, axes = plt.subplots(
        n_subplots, 1,
        figsize=(14, 2.0 * n_subplots),
        sharex=True,
    )
    if n_subplots == 1:
        axes = [axes]

    for ax, (title, s_idx, a_idx) in zip(axes, DIM_GROUPS):
        for i in s_idx:
            ax.plot(frame_index, state[:, i], linewidth=1.0, label="state", color="tab:blue")
        if a_idx is not None:
            for i in a_idx:
                ax.plot(frame_index, action[:, i], linewidth=1.0, label="action", color="tab:orange")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("frame_index")
    fig.suptitle(f"Episode {episode_index} — state & action trajectories", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one episode: plot state & action trajectories jointly."
    )
    parser.add_argument("--dataset_root", type=Path, required=True, help="LeRobot v3.0 dataset root.")
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index to plot.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output dir (default: <dataset_root>/plots).")
    args = parser.parse_args()

    output_dir = args.output_dir or args.dataset_root / "plots"

    frame_index, state, action = load_episode(args.dataset_root, args.episode_index)

    out_path = output_dir / f"episode_{args.episode_index:03d}.png"
    plot_episode(frame_index, state, action, args.episode_index, out_path)

    print(f"Episode {args.episode_index}: {len(frame_index)} frames")
    print(f"  state  shape: {state.shape}")
    print(f"  action shape: {action.shape}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
