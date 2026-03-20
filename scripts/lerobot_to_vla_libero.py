#!/usr/bin/env python3
"""Convert a LeRobot dataset (v3 API) to MY_BaseLiberoDataset on-disk format.

Target format is the one consumed by:
`prismatic/vla/datasets/datasets.py::MY_BaseLiberoDataset`

Example:
python VLA-Adapter/lerobot_convert/lerobot_to_vla_libero.py \
    --repo_id miku112/piper_pick_banana_100 \
    --source_root datasets/lerobot/miku112/piper_pick_banana_100 \
    --output_root datasets/dream-adapter/miku112 \
    --output_dataset_name piper_pick_banana_100_resize_476_converted \
    --use_delta_action True \
    --image_process_type adaptive_resize \
    --num_workers 4 --image_size "[476,476]"
"""

import concurrent.futures
import json
import logging
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import draccus
import h5py
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

LOGGER = logging.getLogger("lerobot_to_vla_libero")


@dataclass
class ConversionConfig:
    repo_id: str = "miku112/piper_pick_banana_100"
    # LeRobot dataset root directory (should contain 'data' and 'meta' subdirectories)
    source_root: Path = Path("datasets/lerobot/miku112/piper_pick_banana_100")
    output_root: Path = Path("datasets/dream-adapter/miku112")
    output_dataset_name: str = "piper_pick_banana_100_converted"
    # lerobot repo may not be pip-installable, so we allow passing a custom path for importing lerobot code
    lerobot_src: Path | None = Path("/home/lxx/repo/lerobot/src")
    # optional episode filter (e.g. [0,1,2] or None for all)
    episodes: list[int] | None = None
    # key 
    primary_camera_key: Optional[str] = "observation.images.image"
    wrist_camera_key: Optional[str] = "observation.images.wrist_image"
    action_key: str = "action"
    proprio_key: str = "observation.state"
    task_key: str = "task"    
    
    use_delta_action: bool = True
    # If enabled, map the last action dimension (gripper) to binary using threshold:
    # value >= threshold is 1, otherwise 0.
    enable_gripper_binary_mapping: bool = False
    gripper_binary_threshold: float = 0.5
    # None: no image processing. Options: center_crop, adaptive_resize.
    image_process_type: str | None = None
    # Unified target resolution; supports 1 value (square) or 2 values (h w).
    image_size: list[int] | None = None
    # if torchcodec is available, it will automatically be used though video_backend is be set as "None"
    video_backend: str | None = None
    overwrite: bool = False
    jpeg_quality: int = 95
    stats_sample_limit: int = 200_000
    num_workers: int = 1
    log_level: str = "INFO"

_WORKER_STATE: dict[str, Any] = {}

def _reservoir_push_bounded(reservoir: list[np.ndarray], sample: np.ndarray, limit: int) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(sample)
        return
    idx = random.randint(0, len(reservoir) - 1)
    reservoir[idx] = sample

def _init_worker(cfg: ConversionConfig, key_map: dict[str, str], output_dataset_dir: Path) -> None:
    converter = LeRobotToVLABaseLiberoConverter(cfg)
    converter.output_dataset_dir = output_dataset_dir
    converter.episodes_dir = output_dataset_dir / "episodes"
    converter.data_info_dir = output_dataset_dir / "data_info"
    dataset = converter._load_lerobot_dataset()
    _WORKER_STATE["converter"] = converter
    _WORKER_STATE["dataset"] = dataset
    _WORKER_STATE["key_map"] = key_map

def _convert_episode_worker(payload: dict[str, Any]) -> dict[str, Any]:
    converter: LeRobotToVLABaseLiberoConverter = _WORKER_STATE["converter"]
    dataset = _WORKER_STATE["dataset"]
    key_map: dict[str, str] = _WORKER_STATE["key_map"]

    episode_idx = int(payload["episode_idx"])
    frame_rows: list[tuple[int, int]] = payload["frame_rows"]

    action_samples: list[np.ndarray] = []
    proprio_samples: list[np.ndarray] = []

    worker_limit = max(1, math.ceil(converter.cfg.stats_sample_limit / max(1, converter.cfg.num_workers)))

    for dataset_idx, expected_frame_idx in frame_rows:
        item = dataset[dataset_idx]
        frame_idx = converter._to_python_int(item["frame_index"])
        if frame_idx != expected_frame_idx:
            LOGGER.warning(
                "Frame index mismatch for episode %d: expected=%d actual=%d",
                episode_idx,
                expected_frame_idx,
                frame_idx,
            )

        step_dir = converter.episodes_dir / f"{episode_idx:06d}" / "steps" / f"{frame_idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        action_source_key = str(key_map["action_source"])
        raw_action = converter._to_1d_float32(item[action_source_key], name=action_source_key)

        action = converter._compute_action(raw_action=raw_action, episode_idx=episode_idx, frame_idx=frame_idx)

        proprio = converter._to_1d_float32(item[key_map["proprio"]], name=key_map["proprio"])
        language_instruction = converter._extract_language(item, key_map["task"])

        primary_image = converter._to_uint8_hwc(item[key_map["primary"]])
        wrist_image = converter._to_uint8_hwc(item[key_map["wrist"]])

        primary_image = converter._process_image(primary_image)
        wrist_image = converter._process_image(wrist_image)

        _reservoir_push_bounded(action_samples, action, worker_limit)
        _reservoir_push_bounded(proprio_samples, proprio, worker_limit)

        converter._write_h5(
            fpath=step_dir / "data.h5",
            action=action,
            proprio=proprio,
            language_instruction=language_instruction,
            dataset_name=converter.cfg.output_dataset_name,
        )
        converter._write_jpg(step_dir / "image_primary.jpg", primary_image)
        converter._write_jpg(step_dir / "image_wrist.jpg", wrist_image)

    return {
        "episode_idx": episode_idx,
        "length": len(frame_rows),
        "action_samples": action_samples,
        "proprio_samples": proprio_samples,
    }

class LeRobotToVLABaseLiberoConverter:
    def __init__(self, cfg: ConversionConfig) -> None:
        self.cfg = cfg
        self.output_dataset_dir = cfg.output_root / cfg.output_dataset_name
        self.episodes_dir = self.output_dataset_dir / "episodes"
        self.data_info_dir = self.output_dataset_dir / "data_info"

        self._action_samples: list[np.ndarray] = []
        self._proprio_samples: list[np.ndarray] = []
        self._prev_action_by_episode: dict[int, np.ndarray] = {}

        self._image_process_mode = self._normalize_image_mode(self.cfg.image_process_type)
        self._target_image_size = self._normalize_image_size(self.cfg.image_size)

        if self._image_process_mode is not None and self._target_image_size is None:
            raise ValueError(
                "image_size must be provided when image_process_type is set "
                "(center_crop or adaptive_resize)."
            )

    def run(self) -> None:
        self._prepare_output_dir()
        dataset = self._load_lerobot_dataset()
        # key map
        key_map = self._resolve_feature_keys(dataset)
        # len(dataset) return the number of frames
        LOGGER.info(
            "Start conversion: frames=%d workers=%d",
            len(dataset),
            max(1, int(self.cfg.num_workers)),
        )

        if int(self.cfg.num_workers) <= 1:
            episode_lengths = self._run_single_process(dataset=dataset, key_map=key_map)
        else:
            episode_lengths = self._run_multi_process(dataset=dataset, key_map=key_map)

        stats_payload = self._build_dataset_statistics(dataset, key_map, episode_lengths)
        self._write_data_info_files(episode_lengths=episode_lengths, stats_payload=stats_payload, key_map=key_map)

        LOGGER.info("Conversion completed. Output: %s", self.output_dataset_dir)

    def _run_single_process(self, *, dataset: Any, key_map: dict[str, str]) -> dict[int, int]:
        episode_lengths: dict[int, int] = {}
        for idx in tqdm(range(len(dataset)), desc="Converting frames"):
            item = dataset[idx]

            episode_idx = self._to_python_int(item["episode_index"])
            frame_idx = self._to_python_int(item["frame_index"])
            episode_lengths[episode_idx] = max(episode_lengths.get(episode_idx, 0), frame_idx + 1)

            self._convert_single_frame(item=item, key_map=key_map, episode_idx=episode_idx, frame_idx=frame_idx)
        return episode_lengths

    def _run_multi_process(self, *, dataset: Any, key_map: dict[str, str]) -> dict[int, int]:
        workers = max(1, int(self.cfg.num_workers))
        episode_to_rows: dict[int, list[tuple[int, int]]] = {}

        # Read index columns directly to avoid triggering expensive image/video decode in dataset.__getitem__.
        if hasattr(dataset, "hf_dataset"):
            episode_col = dataset.hf_dataset["episode_index"]
            frame_col = dataset.hf_dataset["frame_index"]
            iterator = zip(episode_col, frame_col)
            for idx, (episode_val, frame_val) in enumerate(tqdm(iterator, total=len(dataset), desc="Indexing frames")):
                episode_idx = self._to_python_int(episode_val)
                frame_idx = self._to_python_int(frame_val)
                episode_to_rows.setdefault(episode_idx, []).append((idx, frame_idx))
        else:
            for idx in tqdm(range(len(dataset)), desc="Indexing frames"):
                item = dataset[idx]
                episode_idx = self._to_python_int(item["episode_index"])
                frame_idx = self._to_python_int(item["frame_index"])
                episode_to_rows.setdefault(episode_idx, []).append((idx, frame_idx))

        payloads: list[dict[str, Any]] = []
        episode_lengths: dict[int, int] = {}
        for episode_idx, rows in episode_to_rows.items():
            rows.sort(key=lambda x: x[1])
            payloads.append({"episode_idx": episode_idx, "frame_rows": rows})
            episode_lengths[episode_idx] = len(rows)

        LOGGER.info("Dispatching %d episodes across %d processes", len(payloads), workers)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(self.cfg, key_map, self.output_dataset_dir),
        ) as executor:
            futures = [executor.submit(_convert_episode_worker, payload) for payload in payloads]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Converting episodes",
            ):
                result = future.result()
                for sample in result["action_samples"]:
                    self._reservoir_push(self._action_samples, sample)
                for sample in result["proprio_samples"]:
                    self._reservoir_push(self._proprio_samples, sample)

        return episode_lengths

    def _prepare_output_dir(self) -> None:
        if self.output_dataset_dir.exists():
            if not self.cfg.overwrite:
                raise FileExistsError(
                    f"Output dataset already exists: {self.output_dataset_dir}. Use --overwrite to replace it."
                )
            LOGGER.warning("Removing existing output directory: %s", self.output_dataset_dir)
            shutil.rmtree(self.output_dataset_dir)

        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.data_info_dir.mkdir(parents=True, exist_ok=True)

    def _load_lerobot_dataset(self):
        if self.cfg.lerobot_src is not None:
            if not self.cfg.lerobot_src.exists():
                raise FileNotFoundError(f"lerobot_src does not exist: {self.cfg.lerobot_src}")
            sys.path.insert(0, str(self.cfg.lerobot_src))

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError as exc:
            missing_module = getattr(exc, "name", None)
            if missing_module and missing_module != "lerobot":
                raise ImportError(
                    "LeRobot source path is visible, but a required dependency is missing: "
                    f"{missing_module}. Install lerobot runtime dependencies in the current environment."
                ) from exc
            raise ImportError(
                "Failed to import LeRobotDataset. Check --lerobot-src points to lerobot/src and that"
                " the lerobot package is present under that path."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "Failed to import LeRobotDataset. Install lerobot dependencies and/or pass --lerobot-src. "
                f"Original error: {exc.__class__.__name__}: {exc}"
            ) from exc

        ds = LeRobotDataset(
            repo_id=self.cfg.repo_id,
            root=str(self.cfg.source_root),
            episodes=self.cfg.episodes,
            download_videos=False,
            video_backend=self.cfg.video_backend,
        )
        return ds

    def _resolve_feature_keys(self, dataset: Any) -> dict[str, str]:
        camera_keys = list(dataset.meta.camera_keys)
        if not camera_keys:
            raise ValueError("LeRobot dataset has no camera keys; cannot produce image_primary/image_wrist.")

        primary_key = self.cfg.primary_camera_key or camera_keys[0]
        if primary_key not in camera_keys:
            raise ValueError(f"primary_camera_key '{primary_key}' not found in camera_keys={camera_keys}")

        if self.cfg.wrist_camera_key is not None:
            wrist_key = self.cfg.wrist_camera_key
        else:
            wrist_candidates = [k for k in camera_keys if k != primary_key]
            wrist_key = wrist_candidates[0] if wrist_candidates else primary_key

        if wrist_key not in camera_keys:
            raise ValueError(f"wrist_camera_key '{wrist_key}' not found in camera_keys={camera_keys}")

        features = dataset.features
        action_source_key = self.cfg.action_key
        for required in (action_source_key, self.cfg.proprio_key):
            if required not in features:
                raise KeyError(f"Required feature '{required}' not in dataset features: {list(features.keys())}")

        key_map = {
            "primary": primary_key,
            "wrist": wrist_key,
            "action_source": action_source_key,
            "proprio": self.cfg.proprio_key,
            "task": self.cfg.task_key,
        }
        LOGGER.info("Resolved key mapping: %s", key_map)
        return key_map

    def _convert_single_frame(
        self,
        *,
        item: dict[str, Any],
        key_map: dict[str, str],
        episode_idx: int,
        frame_idx: int,
    ) -> None:
        # Create step directory: episodes/000000/steps/0000
        step_dir = self.episodes_dir / f"{episode_idx:06d}" / "steps" / f"{frame_idx:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        # get key of lerobot dataset
        action_source_key = str(key_map["action_source"])
        raw_action = self._to_1d_float32(item[action_source_key], name=action_source_key)
        action = self._compute_action(raw_action=raw_action, episode_idx=episode_idx, frame_idx=frame_idx)
        proprio = self._to_1d_float32(item[key_map["proprio"]], name=key_map["proprio"])
        language_instruction = self._extract_language(item, key_map["task"])

        primary_image = self._to_uint8_hwc(item[key_map["primary"]])
        wrist_image = self._to_uint8_hwc(item[key_map["wrist"]])

        primary_image = self._process_image(primary_image)
        wrist_image = self._process_image(wrist_image)

        self._reservoir_push(self._action_samples, action)
        self._reservoir_push(self._proprio_samples, proprio)

        self._write_h5(
            fpath=step_dir / "data.h5",
            action=action,
            proprio=proprio,
            language_instruction=language_instruction,
            dataset_name=self.cfg.output_dataset_name,
        )

        self._write_jpg(step_dir / "image_primary.jpg", primary_image)
        self._write_jpg(step_dir / "image_wrist.jpg", wrist_image)

    def _build_dataset_statistics(
        self,
        dataset: Any,
        key_map: dict[str, str],
        episode_lengths: dict[int, int],
    ) -> dict[str, Any]:
        source_stats = dataset.meta.stats

        action_stats = self._build_stats_vector(
            source_stats=source_stats,
            key=key_map["action_source"],
            samples=self._action_samples,
            use_source_stats=not self.cfg.use_delta_action,
        )
        proprio_stats = self._build_stats_vector(
            source_stats=source_stats,
            key=key_map["proprio"],
            samples=self._proprio_samples,
        )

        action_mask = None
        if not self.cfg.use_delta_action:
            action_mask = self._extract_mask(source_stats, key_map["action_source"])
        if action_mask is None:
            action_mask = self._infer_action_mask(self._action_samples)
        proprio_mask = self._extract_mask(source_stats, key_map["proprio"])
        if proprio_mask is None:
            proprio_mask = [True] * int(proprio_stats["q01"].shape[0])

        num_transitions = int(sum(episode_lengths.values()))
        num_trajectories = int(len(episode_lengths))

        return {
            self.cfg.output_dataset_name: {
                "action": {
                    "mean": action_stats["mean"].tolist(),
                    "std": action_stats["std"].tolist(),
                    "max": action_stats["max"].tolist(),
                    "min": action_stats["min"].tolist(),
                    "q01": action_stats["q01"].tolist(),
                    "q99": action_stats["q99"].tolist(),
                    "mask": action_mask,
                },
                "proprio": {
                    "mean": proprio_stats["mean"].tolist(),
                    "std": proprio_stats["std"].tolist(),
                    "max": proprio_stats["max"].tolist(),
                    "min": proprio_stats["min"].tolist(),
                    "q01": proprio_stats["q01"].tolist(),
                    "q99": proprio_stats["q99"].tolist(),
                    "mask": proprio_mask,
                },
                "num_transitions": num_transitions,
                "num_trajectories": num_trajectories,
            }
        }

    def _write_data_info_files(
        self,
        *,
        episode_lengths: dict[int, int],
        stats_payload: dict[str, Any],
        key_map: dict[str, str],
    ) -> None:
        self.data_info_dir.mkdir(parents=True, exist_ok=True)

        episode_info = [[f"{ep_idx:06d}", length] for ep_idx, length in sorted(episode_lengths.items())]

        for ep_idx, length in sorted(episode_lengths.items()):
            ep_dir = self.episodes_dir / f"{ep_idx:06d}"
            with h5py.File(ep_dir / "step_info.h5", "w") as f:
                f.create_dataset("length", data=int(length))

        with h5py.File(self.output_dataset_dir / "episodes_info.h5", "w") as f:
            f.create_dataset("num_episodes", data=int(len(episode_lengths)))

        dataset_info_json = self.data_info_dir / f"{self.cfg.output_dataset_name}.json"
        dataset_info_json.parent.mkdir(parents=True, exist_ok=True)
        with open(dataset_info_json, "w", encoding="utf-8") as f:
            json.dump(episode_info, f, ensure_ascii=False, indent=2)

        with open(self.data_info_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, ensure_ascii=False, indent=2)

        manifest = {
            "source": {
                "repo_id": self.cfg.repo_id,
                "source_root": str(self.cfg.source_root),
                "episodes": self.cfg.episodes,
            },
            "target": {
                "dataset_name": self.cfg.output_dataset_name,
                "output_dir": str(self.output_dataset_dir),
            },
            "key_mapping": key_map,
            "converter_config": {
                k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()
            },
        }
        with open(self.data_info_dir / "conversion_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _build_stats_vector(
        self,
        *,
        source_stats: dict[str, Any] | None,
        key: str,
        samples: list[np.ndarray],
        use_source_stats: bool = True,
    ) -> dict[str, np.ndarray]:
        empirical = self._compute_vector_stats(samples)
        if not use_source_stats:
            return empirical
        fields = ("mean", "std", "max", "min", "q01", "q99")

        result: dict[str, np.ndarray] = {}
        for field in fields:
            from_source = self._extract_vector_field(source_stats, key, field)
            if from_source is not None:
                result[field] = from_source
                continue
            if field not in empirical:
                raise RuntimeError(f"Unable to build stats field '{field}' for key '{key}'.")
            LOGGER.warning("Missing %s/%s in source stats, fallback to empirical estimate.", key, field)
            result[field] = empirical[field]
        return result

    @staticmethod
    def _extract_vector_field(source_stats: dict[str, Any] | None, key: str, field: str) -> np.ndarray | None:
        if source_stats is None or key not in source_stats:
            return None
        stats = source_stats[key]
        if field not in stats:
            return None
        arr = np.asarray(stats[field], dtype=np.float32).reshape(-1)
        return arr

    @staticmethod
    def _extract_mask(source_stats: dict[str, Any] | None, key: str) -> list[bool] | None:
        if source_stats is None or key not in source_stats:
            return None
        stats = source_stats[key]
        if "mask" not in stats:
            return None
        arr = np.asarray(stats["mask"]).reshape(-1)
        return [bool(x) for x in arr.tolist()]

    @staticmethod
    def _compute_vector_stats(samples: list[np.ndarray]) -> dict[str, np.ndarray]:
        if not samples:
            raise RuntimeError("No samples available for empirical statistics.")
        matrix = np.stack(samples, axis=0).astype(np.float32)
        return {
            "mean": matrix.mean(axis=0),
            "std": matrix.std(axis=0),
            "max": matrix.max(axis=0),
            "min": matrix.min(axis=0),
            "q01": np.quantile(matrix, 0.01, axis=0),
            "q99": np.quantile(matrix, 0.99, axis=0),
        }

    @staticmethod
    def _infer_action_mask(samples: list[np.ndarray]) -> list[bool]:
        if not samples:
            return []

        matrix = np.stack(samples, axis=0).astype(np.float32)
        mask: list[bool] = []
        for idx in range(matrix.shape[1]):
            col = matrix[:, idx]
            rounded = np.round(col)
            close_to_int = np.isclose(col, rounded, atol=1e-6)
            uniq = np.unique(rounded[close_to_int]) if close_to_int.any() else np.array([], dtype=np.float32)
            discrete = bool(close_to_int.all() and uniq.size <= 3 and set(uniq.tolist()).issubset({-1.0, 0.0, 1.0}))
            mask.append(not discrete)
        return mask

    def _reservoir_push(self, reservoir: list[np.ndarray], sample: np.ndarray) -> None:
        limit = self.cfg.stats_sample_limit
        if len(reservoir) < limit:
            reservoir.append(sample)
            return

        idx = random.randint(0, len(reservoir) - 1)
        reservoir[idx] = sample

    def _extract_language(self, item: dict[str, Any], task_key: str) -> str:
        value = item.get(task_key)
        if value is None:
            value = item.get("task", "")

        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _to_python_int(value: Any) -> int:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    @staticmethod
    def _to_1d_float32(value: Any, *, name: str) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
            value = value.numpy()

        array = np.asarray(value, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim != 1:
            raise ValueError(f"Expected 1D vector for '{name}', got shape={array.shape}")
        return array

    @staticmethod
    def _to_uint8_hwc(value: Any) -> np.ndarray:
        if isinstance(value, Image.Image):
            arr = np.asarray(value)
        elif hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        elif hasattr(value, "numpy") and not isinstance(value, np.ndarray):
            arr = value.numpy()
        else:
            arr = np.asarray(value)

        arr = np.squeeze(arr)

        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim != 3:
            raise ValueError(f"Image must be 3D, got shape={arr.shape}")

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    def _compute_action(self, *, raw_action: np.ndarray, episode_idx: int, frame_idx: int) -> np.ndarray:
        if not self.cfg.use_delta_action:
            action = raw_action
            return self._apply_gripper_binary_mapping(action=action, raw_action=raw_action)

        # Delta is defined between consecutive absolute actions from LeRobot:
        # delta_t = action_t - action_{t-1}. For the first observed frame in
        # each episode, no previous action exists, so delta is zero.
        if frame_idx == 0 or episode_idx not in self._prev_action_by_episode:
            self._prev_action_by_episode[episode_idx] = raw_action.copy()
            if frame_idx != 0:
                LOGGER.warning(
                    "Episode %d first action missing at frame 0; using frame %d as sequence start.",
                    episode_idx,
                    frame_idx,
                )
            action = np.zeros_like(raw_action, dtype=np.float32)
            return self._apply_gripper_binary_mapping(action=action, raw_action=raw_action)

        delta_action = raw_action - self._prev_action_by_episode[episode_idx]
        self._prev_action_by_episode[episode_idx] = raw_action.copy()
        return self._apply_gripper_binary_mapping(action=delta_action, raw_action=raw_action)

    def _apply_gripper_binary_mapping(self, *, action: np.ndarray, raw_action: np.ndarray) -> np.ndarray:
        if not self.cfg.enable_gripper_binary_mapping:
            return action
        if action.ndim != 1 or raw_action.ndim != 1:
            raise ValueError(
                f"Expected 1D action vectors for gripper mapping, got action={action.shape}, raw_action={raw_action.shape}"
            )
        if action.shape[0] == 0 or raw_action.shape[0] == 0:
            raise ValueError("Action vector is empty; cannot map gripper dimension.")

        mapped_action = action.copy()
        threshold = float(self.cfg.gripper_binary_threshold)
        mapped_action[-1] = np.float32(1.0 if float(raw_action[-1]) >= threshold else 0.0)
        return mapped_action

    @staticmethod
    def _normalize_image_mode(mode: str | None) -> str | None:
        if mode is None:
            return None

        text = str(mode).strip().lower()
        if text in {"", "none", "false", "0", "no", "off"}:
            return None
        return text

    @staticmethod
    def _normalize_image_size(raw_size: Any | None) -> tuple[int, int] | None:
        if raw_size is None:
            return None

        if isinstance(raw_size, str):
            cleaned = raw_size.strip().lower().replace("x", " ").replace(",", " ")
            parts = [p for p in cleaned.split() if p]
            if len(parts) == 1:
                values = [int(parts[0])]
            elif len(parts) == 2:
                values = [int(parts[0]), int(parts[1])]
            else:
                raise ValueError(f"Invalid image_size string: {raw_size}")
        elif isinstance(raw_size, int):
            values = [raw_size]
        elif isinstance(raw_size, Sequence):
            values = [int(v) for v in raw_size]
        else:
            raise ValueError(f"Unsupported image_size type: {type(raw_size)}")

        if len(values) == 1:
            h = w = values[0]
        elif len(values) == 2:
            h, w = values
        else:
            raise ValueError(f"image_size expects 1 or 2 values, got: {values}")

        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image_size: {(h, w)}")
        return int(h), int(w)

    def _process_image(self, arr: np.ndarray) -> np.ndarray:
        if self._image_process_mode is None:
            return arr
        if self._target_image_size is None:
            raise ValueError("image_size is required when image processing is enabled.")

        target_h, target_w = self._target_image_size
        image = Image.fromarray(arr)
        target_size_wh = (target_w, target_h)

        if self._image_process_mode == "center_crop":
            processed = ImageOps.fit(
                image,
                size=target_size_wh,
                method=Image.BILINEAR,
                centering=(0.5, 0.5),
            )
        elif self._image_process_mode == "adaptive_resize":
            processed = ImageOps.pad(
                image,
                size=target_size_wh,
                method=Image.BILINEAR,
                color=(255, 255, 255),
                centering=(0.5, 0.5),
            )
        else:
            raise ValueError(f"Unsupported image process mode: {self._image_process_mode}")

        return np.asarray(processed)

    def _write_h5(
        self,
        *,
        fpath: Path,
        action: np.ndarray,
        proprio: np.ndarray,
        language_instruction: str,
        dataset_name: str,
    ) -> None:
        with h5py.File(fpath, "w") as f:
            f.create_dataset("action", data=action)
            obs = f.create_group("observation")
            obs.create_dataset("proprio", data=proprio)
            f.create_dataset("language_instruction", data=np.bytes_(language_instruction))
            f.create_dataset("dataset_name", data=np.bytes_(dataset_name))

    def _write_jpg(self, fpath: Path, arr: np.ndarray) -> None:
        image = Image.fromarray(arr)
        image.save(fpath, format="JPEG", quality=self.cfg.jpeg_quality)

def main() -> None:
    cfg = draccus.parse(config_class=ConversionConfig)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Print configuration for verification
    print("-" * 20 + " Conversion Configuration " + "-" * 20)
    print(draccus.dump(cfg))
    print("-" * 65)

    converter = LeRobotToVLABaseLiberoConverter(cfg)
    converter.run()

if __name__ == "__main__":
    main()
