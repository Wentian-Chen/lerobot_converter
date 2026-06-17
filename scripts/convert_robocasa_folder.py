"""Convert RoboCasa all_episodes HDF5 files to one LeRobot v3.0 dataset using multi-process.

Recursively scans for HDF5 files, filters by --success=True in filename,
distributes evenly across workers, converts each to a temp LeRobot dataset,
then merges via aggregate_datasets().

Usage:
    # Convert all tasks
    .venv/bin/python scripts/convert_robocasa_folder.py \
        --source_dir /data2/repo/datasets/RoboCasa-Cosmos-Policy/all_episodes \
        --output_dir /data2/repo/datasets/RoboCasa-Cosmos-Policy/lerobot \
        --num_workers 8

    # Convert single task
    .venv/bin/python scripts/convert_robocasa_folder.py \
        --source_dir /data2/repo/datasets/RoboCasa-Cosmos-Policy/all_episodes/CloseDoubleDoor \
        --output_dir /data2/repo/datasets/RoboCasa-Cosmos-Policy/lerobot_close_double_door \
        --num_workers 4
"""

import io
import logging
import re
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import draccus
import h5py
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

LEROBOT_SRC_DIR = PROJECT_ROOT.parent / "lerobot" / "src"
if str(LEROBOT_SRC_DIR) not in sys.path:
    sys.path.append(str(LEROBOT_SRC_DIR))

from lerobot_converter.hdf5_adapter import Hdf5ToLeRobotConverter
from lerobot_converter.models import ConversionOptions
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.utils import INFO_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

FPS = 20

FEATURES: dict[str, dict[str, Any]] = {
    "observation.images.primary": {
        "dtype": "video",
        "shape": (3, 224, 224),
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 224,
            "video.width": 224,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.secondary": {
        "dtype": "video",
        "shape": (3, 224, 224),
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 224,
            "video.width": 224,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (3, 224, 224),
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 224,
            "video.width": 224,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float64",
        "shape": (9,),
        "names": [
            "gripper_0", "gripper_1",
            "pos_x", "pos_y", "pos_z",
            "quat_x", "quat_y", "quat_z", "quat_w",
        ],
    },
    "action": {
        "dtype": "float64",
        "shape": (12,),
    },
}

# ── Helpers ────────────────────────────────────────────────────────────────


def decode_jpeg(jpeg_bytes: np.ndarray) -> np.ndarray:
    """JPEG bytes -> (H, W, 3) uint8 numpy array."""
    return np.array(Image.open(io.BytesIO(jpeg_bytes)))


def is_success_file(filename: str) -> bool:
    """Parse --success=True/False from filename."""
    m = re.search(r'--success=(True|False)--', filename)
    if not m:
        return False
    return m.group(1) == "True"


def parse_task_from_filename(filename: str) -> str:
    """Parse task description from filename.

    episode_data--task=close_the_cabinet_doors--2025-10-02_15-20-39--ep=2--success=True--regen_demo.hdf5
      -> 'close the cabinet doors'
    """
    m = re.search(r'--task=(.+?)--', filename)
    return m.group(1).replace("_", " ") if m else "unknown"


def collect_hdf5_files(source_dir: Path) -> list[Path]:
    """Recursively collect all HDF5 files with --success=True in filename."""
    hdf5_files = []
    for f in source_dir.rglob("*.hdf5"):
        if is_success_file(f.name):
            hdf5_files.append(f)
    return sorted(hdf5_files)


def distribute_files(files: list[Path], num_workers: int) -> list[list[Path]]:
    """Distribute files evenly across workers."""
    chunks: list[list[Path]] = [[] for _ in range(num_workers)]
    for i, f in enumerate(files):
        chunks[i % num_workers].append(f)
    return [c for c in chunks if c]  # Remove empty chunks


def discover_datasets(target_root: Path) -> tuple[list[str], list[Path]]:
    """Scan subdirectories for meta/info.json, return (repo_ids, roots)."""
    repo_ids: list[str] = []
    roots: list[Path] = []
    for child in sorted(target_root.iterdir()):
        if not child.is_dir():
            continue
        info_path = child / INFO_PATH
        if info_path.exists():
            repo_ids.append(child.name)
            roots.append(child.resolve())
            log.info("Found dataset: %s", child.name)
        else:
            log.debug("Skipping non-dataset directory: %s", child.name)
    return repo_ids, roots


# ── Converter ──────────────────────────────────────────────────────────────


class RobocasaSingleEpisodeHdf5Converter(Hdf5ToLeRobotConverter):
    """HDF5 adapter for RoboCasa all_episodes format.

    Each file contains one episode with top-level datasets:
    - actions (T, 12)
    - proprio (T, 9)
    - primary_images_jpeg (T,)
    - secondary_images_jpeg (T,)
    - wrist_images_jpeg (T,)
    """

    def iter_source_episodes(
        self,
        source: str | Path,
        options: ConversionOptions,
    ) -> Iterable[dict[str, Any]]:
        source_path = Path(source)
        file_paths = self._resolve_hdf5_files(source_path)
        if not file_paths:
            raise FileNotFoundError(f"No HDF5 files found: {source_path}")

        def _generator() -> Iterable[dict[str, Any]]:
            for episode_id, file_path in enumerate(file_paths):
                task = parse_task_from_filename(file_path.name)
                with h5py.File(file_path, "r") as f:
                    yield self.extract_episode_from_file(
                        file_obj=f,
                        file_path=file_path,
                        episode_id=episode_id,
                        options=options,
                        task=task,
                    )

        return _generator()

    def extract_episode_from_file(
        self,
        file_obj: Any,
        file_path: Path,
        episode_id: int,
        options: ConversionOptions,
        task: str = "",
    ) -> dict[str, Any]:
        _ = file_path
        if options.features is None:
            raise ValueError("ConversionOptions.features is required.")

        feature_keys = set(options.features.keys())

        actions = np.array(file_obj["actions"])             # (T, 12)
        proprio = np.array(file_obj["proprio"])             # (T, 9)
        primary_jpeg = file_obj["primary_images_jpeg"]      # (T,)
        secondary_jpeg = file_obj["secondary_images_jpeg"]  # (T,)
        wrist_jpeg = file_obj["wrist_images_jpeg"]          # (T,)

        T = actions.shape[0]
        assert proprio.shape == (T, 9), f"proprio shape mismatch: {proprio.shape}"
        assert primary_jpeg.shape == (T,), f"primary_images_jpeg shape mismatch: {primary_jpeg.shape}"
        assert secondary_jpeg.shape == (T,), f"secondary_images_jpeg shape mismatch: {secondary_jpeg.shape}"
        assert wrist_jpeg.shape == (T,), f"wrist_images_jpeg shape mismatch: {wrist_jpeg.shape}"

        steps: list[dict[str, Any]] = []
        for idx in range(T):
            feature_values: dict[str, Any] = {}

            if "observation.state" in feature_keys:
                feature_values["observation.state"] = proprio[idx].astype(np.float64)

            if "action" in feature_keys:
                feature_values["action"] = actions[idx].astype(np.float64)

            if "observation.images.primary" in feature_keys:
                img = decode_jpeg(primary_jpeg[idx])
                feature_values["observation.images.primary"] = np.transpose(img, (2, 0, 1))

            if "observation.images.secondary" in feature_keys:
                img = decode_jpeg(secondary_jpeg[idx])
                feature_values["observation.images.secondary"] = np.transpose(img, (2, 0, 1))

            if "observation.images.wrist" in feature_keys:
                img = decode_jpeg(wrist_jpeg[idx])
                feature_values["observation.images.wrist"] = np.transpose(img, (2, 0, 1))

            missing = feature_keys - set(feature_values.keys())
            if missing:
                raise ValueError(f"Missing feature keys: {sorted(missing)}")

            steps.append({
                "task": task,
                "feature_values": feature_values,
                "timestamp": float(idx) / FPS,
            })

        return {"episode_id": episode_id, "steps": steps}


# ── Worker function ────────────────────────────────────────────────────────


def convert_file_chunk(
    file_paths: list[str],
    task_instructions: list[str],
    temp_output: str,
    fps: int,
    robot_type: str,
    result_queue: Queue,
) -> None:
    """Convert all HDF5 files in chunk to ONE temp LeRobot dataset. Runs in a worker process."""
    temp_output = Path(temp_output)

    log.info("Worker converting %d files to %s", len(file_paths), temp_output.name)

    options = ConversionOptions(
        dataset_name=temp_output.name,
        fps=fps,
        robot_type=robot_type,
        use_videos=True,
        features=FEATURES,
        default_task=task_instructions[0],
    )

    adapter = RobocasaSingleEpisodeHdf5Converter()

    # First file creates the dataset
    adapter.convert(file_paths[0], str(temp_output), options)

    # Remaining files add to the same dataset
    for file_path, task in zip(file_paths[1:], task_instructions[1:]):
        adapter.convert(file_path, str(temp_output), options)

    report = adapter.finalize_target()

    log.info(
        "Worker done %s: %d episodes, %d frames",
        temp_output.name, report.episode_count, report.frame_count,
    )
    result_queue.put({
        "worker": temp_output.name,
        "episodes": report.episode_count,
        "frames": report.frame_count,
    })


# ── Config ─────────────────────────────────────────────────────────────────


@dataclass
class RobocasaFolderConvertConfig:
    source_dir: str = ""
    output_dir: str = ""
    num_workers: int = 4
    fps: int = FPS
    robot_type: str = "panda"
    dataset_name: str = "robocasa"
    temp_dir: str = ""
    clean_temp: bool = True


# ── Entry point ────────────────────────────────────────────────────────────


@draccus.wrap()
def run_robocasa_folder_convert(cfg: RobocasaFolderConvertConfig) -> None:
    if not cfg.source_dir:
        raise ValueError("Provide --source_dir.")
    if not cfg.output_dir:
        raise ValueError("Provide --output_dir.")

    source_dir = Path(cfg.source_dir)
    output_dir = Path(cfg.output_dir)

    # 1. Recursively collect success HDF5 files
    hdf5_files = collect_hdf5_files(source_dir)
    if not hdf5_files:
        raise FileNotFoundError(f"No success HDF5 files found in {source_dir}")
    log.info("Found %d success HDF5 files", len(hdf5_files))

    # 2. Validate num_workers
    if cfg.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {cfg.num_workers}")
    actual_workers = min(cfg.num_workers, len(hdf5_files))
    if actual_workers != cfg.num_workers:
        log.info("Reducing num_workers from %d to %d (file count)", cfg.num_workers, actual_workers)

    # 3. Distribute files evenly across workers
    file_chunks = distribute_files(hdf5_files, actual_workers)
    log.info("Distributed files: %d chunks, sizes: %s", len(file_chunks), [len(c) for c in file_chunks])

    # 4. Setup temp directory
    temp_dir = Path(cfg.temp_dir) if cfg.temp_dir else output_dir.parent / f".tmp_{output_dir.name}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 5. Build worker args (one args tuple per worker, not per file)
    task_args: list[tuple[list[str], list[str], str, int, str, Queue]] = []
    result_queue: Queue = Queue()

    for worker_id, chunk in enumerate(file_chunks):
        file_paths = [str(f) for f in chunk]
        tasks = [parse_task_from_filename(f.name) for f in chunk]
        temp_subdir = temp_dir / f"worker_{worker_id}"
        task_args.append((
            file_paths,
            tasks,
            str(temp_subdir),
            cfg.fps,
            cfg.robot_type,
            result_queue,
        ))

    # 6. Launch worker processes (one process per worker)
    log.info("Launching %d workers...", actual_workers)

    processes: list[Process] = []
    for args in task_args:
        p = Process(
            target=convert_file_chunk,
            args=args,
            name=f"Worker-{args[2].split('/')[-1]}",
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results: list[dict[str, Any]] = []
    while not result_queue.empty():
        results.append(result_queue.get())

    total_episodes = sum(r["episodes"] for r in results)
    total_frames = sum(r["frames"] for r in results)
    log.info("All workers done: %d episodes, %d frames total", total_episodes, total_frames)

    # 7. Merge temp datasets using standard discover + aggregate
    log.info("Merging %d temp datasets...", len(results))
    repo_ids, roots = discover_datasets(temp_dir)
    if not repo_ids:
        raise RuntimeError("No valid temp datasets found for merging.")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=cfg.dataset_name,
        roots=roots,
        aggr_root=output_dir,
    )
    log.info("Merged dataset saved to %s", output_dir)

    # 8. Clean temp dirs
    if cfg.clean_temp:
        log.info("Cleaning temp dir %s", temp_dir)
        shutil.rmtree(temp_dir)

    # 9. Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Source:     {source_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Episodes:   {total_episodes}")
    print(f"  Frames:     {total_frames}")
    print(f"  Workers:    {actual_workers}")


if __name__ == "__main__":
    run_robocasa_folder_convert()
