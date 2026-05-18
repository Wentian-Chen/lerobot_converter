"""
uv run scripts/aggregate_lerobot_datasets_from_path.py \
    --target_root /home/charles/workspaces/Double_Piper_Teleop/datasets_all \
    --aggr_repo_id merged_dataset \
    --aggr_root /home/charles/workspaces/Double_Piper_Teleop/datasets_lerobot/miku112/merged_dataset 
"""
from dataclasses import dataclass, field
from pathlib import Path
import logging
import sys

import draccus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

WORKSPACE_ROOT = PROJECT_ROOT.parent
LEROBOT_SRC_DIR = WORKSPACE_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC_DIR) not in sys.path:
    sys.path.append(str(LEROBOT_SRC_DIR))

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.utils import INFO_PATH


@dataclass
class AggregateFromPathConfig:
    target_root: str = ""
    aggr_repo_id: str = ""
    aggr_root: str = ""
    data_files_size_in_mb: float = field(default=None)
    video_files_size_in_mb: float = field(default=None)
    chunk_size: int = field(default=None)
    log_level: str = "INFO"


def discover_datasets(target_root: Path) -> tuple[list[str], list[Path]]:
    repo_ids: list[str] = []
    roots: list[Path] = []

    for child in sorted(target_root.iterdir()):
        if not child.is_dir():
            continue
        info_path = child / INFO_PATH
        if info_path.exists():
            repo_ids.append(child.name)
            roots.append(child.resolve())
            logging.info(f"Found dataset: {child.name}")
        else:
            logging.debug(f"Skipping non-dataset directory: {child.name}")

    return repo_ids, roots


@draccus.wrap()
def main(cfg: AggregateFromPathConfig):
    if not cfg.target_root:
        raise ValueError("请提供 --target_root。")
    if not cfg.aggr_repo_id:
        raise ValueError("请提供 --aggr_repo_id。")
    if not cfg.aggr_root:
        raise ValueError("请提供 --aggr_root。")

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    target_root = Path(cfg.target_root).resolve()
    if not target_root.is_dir():
        raise ValueError(f"target_root 不存在或不是目录: {target_root}")

    repo_ids, roots = discover_datasets(target_root)

    if not repo_ids:
        raise ValueError(
            f"在 {target_root} 下未找到任何有效的 LeRobot 数据集"
            "（需要子目录中包含 {INFO_PATH}）。"
        )

    logging.info(
        f"共发现 {len(repo_ids)} 个数据集，开始合并为「{cfg.aggr_repo_id}」..."
    )

    aggr_root = Path(cfg.aggr_root).resolve()

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=cfg.aggr_repo_id,
        roots=roots,
        aggr_root=aggr_root,
        data_files_size_in_mb=cfg.data_files_size_in_mb,
        video_files_size_in_mb=cfg.video_files_size_in_mb,
        chunk_size=cfg.chunk_size,
    )


if __name__ == "__main__":
    main()
