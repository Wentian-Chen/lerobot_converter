"""Convert all LIBERO HDF5 files in a folder to one LeRobot v3.0 dataset using multi-process.

Each HDF5 file is converted to a temporary LeRobot dataset by a worker process,
then all temp datasets are merged via aggregate_datasets().

Usage:
    .venv/bin/python scripts/convert_libero_folder.py \
        --source_dir /data2/repo/datasets/LIBERO-Cosmos-Policy/success_only/libero_10_regen \
        --output_dir /data2/repo/datasets/LIBERO-Cosmos-Policy/success_only/libero_10_lerobot \
        --metainfo_path /data2/repo/datasets/LIBERO-Cosmos-Policy/success_only/libero_10_regen/libero_10_metainfo.json \
        --num_workers 5
"""

import io
import json
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
    "observation.images.image": {
        "dtype": "video",
        "shape": (3, 256, 256),
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 256,
            "video.width": 256,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (3, 256, 256),
        "names": ["channels", "height", "width"],
        "info": {
            "video.height": 256,
            "video.width": 256,
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
            "pos_x", "pos_y", "pos_z",
            "quat_x", "quat_y", "quat_z", "quat_w",
            "gripper", "pad",
        ],
    },
    "action": {
        "dtype": "float64",
        "shape": (7,),
        "names": [
            "pos_x", "pos_y", "pos_z",
            "quat_x", "quat_y", "quat_z",
            "gripper",
        ],
    },
}

# Reorder quat (w,x,y,z) → (x,y,z,w)
_QUAT_REORDER = [0, 1, 2, 4, 5, 6, 3, 7, 8]

# ── Helpers ────────────────────────────────────────────────────────────────


def decode_jpeg(jpeg_bytes: np.ndarray) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(jpeg_bytes)))


def task_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    m = re.match(r"^[A-Z_]+SCENE\d+_(.+)$", stem)
    return (m.group(1) if m else stem).replace("_", " ")


def load_metainfo(metainfo_path: Path) -> dict[str, dict]:
    with open(metainfo_path) as f:
        return json.load(f)


def find_metainfo_key(task_readable: str, metainfo_keys: list[str]) -> str | None:
    normalized = task_readable.replace(" ", "_").lower()
    for key in metainfo_keys:
        if key.lower() == normalized:
            return key
    return None


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


# ── Converter (duplicated from tests/test_libero_convert.py) ──────────────


class LiberoMultiDemoHdf5Converter(Hdf5ToLeRobotConverter):
    """HDF5 adapter for LIBERO: each file contains many demo_N groups."""

    def __init__(self, metainfo_path: Path | None = None) -> None:
        super().__init__()
        self._metainfo: dict[str, dict] | None = None
        self._metainfo_keys: list[str] | None = None
        self._metainfo_path = metainfo_path

    def iter_source_episodes(
        self,
        source: str | Path,
        options: ConversionOptions,
    ) -> Iterable[dict[str, Any]]:
        source_path = Path(source)
        file_paths = self._resolve_hdf5_files(source_path)
        if not file_paths:
            raise FileNotFoundError(f"No HDF5 files found: {source_path}")

        if self._metainfo is None and self._metainfo_path is not None:
            self._metainfo = load_metainfo(self._metainfo_path)
            self._metainfo_keys = list(self._metainfo.keys())

        def _generator() -> Iterable[dict[str, Any]]:
            global_episode_id = 0
            for file_path in file_paths:
                task_readable = task_from_filename(file_path.name)
                meta_key = find_metainfo_key(task_readable, self._metainfo_keys or [])
                task = meta_key.replace("_", " ") if meta_key else task_readable

                with h5py.File(file_path, "r") as f:
                    demo_keys = sorted(
                        [k for k in f["data"].keys() if k.startswith("demo_")],
                        key=lambda x: int(x.split("_")[1]),
                    )
                    for demo_key in demo_keys:
                        yield self.extract_episode_from_file(
                            file_obj=f,
                            file_path=file_path,
                            episode_id=global_episode_id,
                            options=options,
                            demo_key=demo_key,
                            task=task,
                        )
                        global_episode_id += 1

        return _generator()

    def extract_episode_from_file(
        self,
        file_obj: Any,
        file_path: Path,
        episode_id: int,
        options: ConversionOptions,
        demo_key: str = "demo_0",
        task: str = "",
    ) -> dict[str, Any]:
        _ = file_path
        if options.features is None:
            raise ValueError("ConversionOptions.features is required.")

        feature_keys = set(options.features.keys())
        demo = file_obj[f"data/{demo_key}"]

        actions = np.array(demo["actions"])
        robot_states = np.array(demo["robot_states"])
        agentview_jpeg = demo["obs/agentview_rgb_jpeg"]
        wrist_jpeg = demo["obs/eye_in_hand_rgb_jpeg"]

        T = actions.shape[0]
        robot_states_reordered = robot_states[:, _QUAT_REORDER]

        steps: list[dict[str, Any]] = []
        for idx in range(T):
            feature_values: dict[str, Any] = {}

            if "observation.state" in feature_keys:
                feature_values["observation.state"] = robot_states_reordered[idx].astype(np.float64)
            if "action" in feature_keys:
                feature_values["action"] = actions[idx].astype(np.float64)
            if "observation.images.image" in feature_keys:
                img = decode_jpeg(agentview_jpeg[idx])
                feature_values["observation.images.image"] = np.transpose(img, (2, 0, 1))
            if "observation.images.wrist_image" in feature_keys:
                img = decode_jpeg(wrist_jpeg[idx])
                feature_values["observation.images.wrist_image"] = np.transpose(img, (2, 0, 1))

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


def resolve_task_from_filename(filename: str, metainfo_keys: list[str]) -> str:
    """Resolve task instruction from filename using metainfo keys."""
    task_readable = task_from_filename(filename)
    meta_key = find_metainfo_key(task_readable, metainfo_keys)
    return meta_key.replace("_", " ") if meta_key else task_readable


def convert_file_chunk(
    file_paths: list[str],
    task_instructions: list[str],
    temp_output: str,
    metainfo_path: str,
    fps: int,
    robot_type: str,
    result_queue: Queue,
) -> None:
    """Convert all HDF5 files in chunk to ONE temp LeRobot dataset. Runs in a worker process."""
    temp_output = Path(temp_output)
    metainfo_path = Path(metainfo_path)

    log.info("Worker converting %d files to %s", len(file_paths), temp_output.name)

    options = ConversionOptions(
        dataset_name=temp_output.name,
        fps=fps,
        robot_type=robot_type,
        use_videos=True,
        features=FEATURES,
        default_task=task_instructions[0],
    )

    adapter = LiberoMultiDemoHdf5Converter(metainfo_path=metainfo_path)

    # First file creates the dataset (contains multiple demos)
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
class LiberoFolderConvertConfig:
    source_dir: str = ""
    output_dir: str = ""
    metainfo_path: str = ""
    num_workers: int = 5
    fps: int = FPS
    robot_type: str = "panda"
    dataset_name: str = "libero_10"
    temp_dir: str = ""
    clean_temp: bool = True


# ── Entry point ────────────────────────────────────────────────────────────


@draccus.wrap()
def run_libero_folder_convert(cfg: LiberoFolderConvertConfig) -> None:
    if not cfg.source_dir:
        raise ValueError("Provide --source_dir.")
    if not cfg.output_dir:
        raise ValueError("Provide --output_dir.")
    if not cfg.metainfo_path:
        raise ValueError("Provide --metainfo_path.")

    source_dir = Path(cfg.source_dir)
    output_dir = Path(cfg.output_dir)
    metainfo_path = Path(cfg.metainfo_path)

    # 1. Scan HDF5 files
    hdf5_files = sorted(source_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {source_dir}")
    log.info("Found %d HDF5 files", len(hdf5_files))

    # 2. Validate num_workers
    if cfg.num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {cfg.num_workers}")
    actual_workers = min(cfg.num_workers, len(hdf5_files))
    if actual_workers != cfg.num_workers:
        log.info("Reducing num_workers from %d to %d (file count)", cfg.num_workers, actual_workers)

    # 3. Load metainfo for task mapping
    metainfo = load_metainfo(metainfo_path)
    metainfo_keys = list(metainfo.keys())

    # 4. Distribute files evenly across workers
    file_chunks = distribute_files(hdf5_files, actual_workers)
    log.info("Distributed files: %d chunks, sizes: %s", len(file_chunks), [len(c) for c in file_chunks])

    # 5. Setup temp directory
    temp_dir = Path(cfg.temp_dir) if cfg.temp_dir else output_dir.parent / f".tmp_{output_dir.name}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 6. Build worker args (one args tuple per worker, not per file)
    task_args: list[tuple[list[str], list[str], str, str, int, str, Queue]] = []
    result_queue: Queue = Queue()

    for worker_id, chunk in enumerate(file_chunks):
        file_paths = [str(f) for f in chunk]
        tasks = [resolve_task_from_filename(f.name, metainfo_keys) for f in chunk]
        temp_subdir = temp_dir / f"worker_{worker_id}"
        task_args.append((
            file_paths,
            tasks,
            str(temp_subdir),
            str(metainfo_path),
            cfg.fps,
            cfg.robot_type,
            result_queue,
        ))

    # 7. Launch worker processes (one process per worker)
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

    # 8. Merge temp datasets using standard discover + aggregate
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

    # 9. Clean temp dirs
    if cfg.clean_temp:
        log.info("Cleaning temp dir %s", temp_dir)
        shutil.rmtree(temp_dir)

    # 10. Summary
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Source:     {source_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Episodes:   {total_episodes}")
    print(f"  Frames:     {total_frames}")
    print(f"  Workers:    {actual_workers}")


if __name__ == "__main__":
    run_libero_folder_convert()
