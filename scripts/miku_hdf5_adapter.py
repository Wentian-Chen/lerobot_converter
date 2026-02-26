from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any

import draccus
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from lerobot_convertor.hdf5_adapter import Hdf5ToLeRobotConvertor
from lerobot_convertor.models import ConversionOptions, DatasetsConvertorConfig


class MikuSingleEpisodeHdf5Convertor(Hdf5ToLeRobotConvertor):
    """Adapter for datasets where each HDF5 file is exactly one episode.

    Expected structure per file:
      /cam_head/color       (T, H, W, 3)
      /cam_head/timestamp   (T,)
      /cam_wrist/color      (T, H, W, 3)
      /cam_wrist/timestamp  (T,)
      /left_arm/joint       (T, 6)
      /left_arm/qpos        (T, 6)
      /left_arm/gripper     (T,)
      /left_arm/timestamp   (T,)
    """

    def __init__(self) -> None:
        super().__init__()
        self._source_root: Path | None = None
        self._task_instructions_cache: list[str] | None = None

    def iter_source_episodes(
        self,
        source: str | Path,
        options: ConversionOptions,
    ):
        source_path = Path(source)
        self._source_root = source_path if source_path.is_dir() else source_path.parent
        self._task_instructions_cache = None
        return super().iter_source_episodes(source, options)

    def extract_episode_from_file(
        self,
        file_obj: Any,
        file_path: Path,
        episode_id: int,
        options: ConversionOptions,
    ) -> dict[str, Any]:
        _ = file_path
        if options.features is None:
            raise ValueError("ConversionOptions.features is required.")

        task = self._select_task_for_episode(options)

        feature_keys = set(options.features.keys())
        cam_head_color = self._read_array(file_obj, "cam_head/color")
        cam_wrist_color = self._read_array(file_obj, "cam_wrist/color")
        left_arm_joint = self._read_array(file_obj, "left_arm/joint")
        left_arm_qpos = self._read_array(file_obj, "left_arm/qpos")
        left_arm_gripper = self._read_array(file_obj, "left_arm/gripper")

        step_count = self._infer_step_count_from_arrays(
            cam_head_color,
            cam_wrist_color,
            left_arm_joint,
            left_arm_qpos,
            left_arm_gripper,
        )

        timestamps_raw = self._first_existing_array(
            file_obj,
            ["left_arm/timestamp", "cam_head/timestamp", "cam_wrist/timestamp"],
            expected_length=step_count,
        )
        timestamps_s = self._normalize_timestamps_to_seconds(timestamps_raw)

        steps: list[dict[str, Any]] = []
        for idx in range(step_count):
            feature_values: dict[str, Any] = {}

            if "observation.state" in feature_keys:
                state = np.concatenate(
                    [left_arm_qpos[idx].astype(np.float64), np.asarray([left_arm_gripper[idx]], dtype=np.float64)],
                    axis=0,
                )
                feature_values["observation.state"] = state

            if "action" in feature_keys:
                action = np.concatenate(
                    [left_arm_joint[idx].astype(np.float64), np.asarray([left_arm_gripper[idx]], dtype=np.float64)],
                    axis=0,
                )
                feature_values["action"] = action

            if "observation.images.image" in feature_keys:
                feature_values["observation.images.image"] = np.transpose(cam_head_color[idx], (2, 0, 1))

            if "observation.images.wrist_image" in feature_keys:
                feature_values["observation.images.wrist_image"] = np.transpose(cam_wrist_color[idx], (2, 0, 1))

            missing = feature_keys - set(feature_values.keys())
            if missing:
                raise ValueError(
                    f"extract_episode_from_file failed to populate feature keys: {sorted(missing)}"
                )

            steps.append(
                {
                    "task": task,
                    "feature_values": feature_values,
                    "timestamp": float(timestamps_s[idx]),
                }
            )

        return {"episode_id": episode_id, "steps": steps}

    def _select_task_for_episode(self, options: ConversionOptions) -> str:
        if not options.augment_task_instruction:
            return options.default_task

        instructions = self._load_task_instructions()
        if not instructions:
            raise ValueError("tasks_instruction.json is empty or has no valid instruction strings.")
        return random.choice(instructions)

    def _load_task_instructions(self) -> list[str]:
        if self._task_instructions_cache is not None:
            return self._task_instructions_cache
        if self._source_root is None:
            raise RuntimeError("Source root is not initialized before loading task instructions.")

        file_path = self._source_root / "tasks_instruction.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"augment_task_instruction=True but file not found: {file_path}"
            )

        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("tasks"), list):
                raw_items = payload["tasks"]
            elif isinstance(payload.get("task_instructions"), list):
                raw_items = payload["task_instructions"]
            else:
                raw_items = []
        else:
            raw_items = []

        instructions = [str(item).strip() for item in raw_items if str(item).strip()]
        self._task_instructions_cache = instructions
        return instructions

    @staticmethod
    def _read_array(file_obj: Any, key: str) -> np.ndarray:
        if key not in file_obj:
            raise ValueError(f"Missing required dataset '{key}'.")
        return np.asarray(file_obj[key])

    @staticmethod
    def _infer_step_count_from_arrays(*arrays: np.ndarray) -> int:
        lengths = {arr.shape[0] for arr in arrays}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent step count across arrays: {lengths}")
        step_count = int(next(iter(lengths)))
        if step_count <= 0:
            raise ValueError("Episode must contain at least one step.")
        return step_count

    @staticmethod
    def _first_existing_array(file_obj: Any, keys: list[str], expected_length: int) -> np.ndarray:
        for key in keys:
            if key in file_obj:
                arr = np.asarray(file_obj[key])
                if arr.shape[0] != expected_length:
                    raise ValueError(
                        f"Timestamp field '{key}' length {arr.shape[0]} does not match {expected_length}."
                    )
                return arr
        return np.arange(expected_length, dtype=np.float64)

    @staticmethod
    def _normalize_timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
        arr = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if arr.size <= 1:
            return np.zeros_like(arr, dtype=np.float64)

        deltas = np.diff(arr)
        positive_deltas = deltas[deltas > 0]
        if positive_deltas.size == 0:
            return np.arange(arr.size, dtype=np.float64)

        median_delta = float(np.median(positive_deltas))
        if median_delta > 1e8:
            scale = 1e-9
        elif median_delta > 1e5:
            scale = 1e-6
        elif median_delta > 1e2:
            scale = 1e-3
        else:
            scale = 1.0

        normalized = (arr - arr[0]) * scale
        return normalized.astype(np.float64)

@dataclass
class MikuHdf5AdapterConfig(DatasetsConvertorConfig):
    source: str = ""
    output_dir: str = ""
    dataset_name: str = ""
    fps: int = 10
    robot_type: str | None = ""
    use_videos: bool = True
    features: dict[str, dict[str, Any]] | None = None
    default_task: str = ""
    augment_task_instruction: bool = False

@draccus.wrap()
def run_miku_hdf5_adapter(cfg: MikuHdf5AdapterConfig):
    if not cfg.source:
        raise ValueError("Provide --source path to HDF5 file/folder.")
    if not cfg.output_dir:
        raise ValueError("Provide --output_dir for converted dataset output.")
    feature =  {
        "observation.images.image": {
            "dtype": "video",
            "shape": (
                3,
                480,
                640,
            ),
            "names": [
                "channels",
                "height",
                "width"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": False
            }
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": (
                3,
                480,
                640,
            ),
            "names": [
                "channels",
                "height",
                "width"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": False
            }
        },
        "observation.state": {
            "dtype": "float64",
            "shape": (
                7,
            ),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper"
            ]
        },
        "action": {
            "dtype": "float64",
            "shape": (
                7,
            ),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper"
            ]
        }
    }
    options = cfg.options
    if options.features is None:
        options = ConversionOptions(
            dataset_name=options.dataset_name,
            fps=options.fps,
            robot_type=options.robot_type,
            use_videos=options.use_videos,
            features=feature,
            default_task=options.default_task,
            augment_task_instruction=options.augment_task_instruction,
            task_key=options.task_key,
            timestamp_key=options.timestamp_key,
        )

    adapter = MikuSingleEpisodeHdf5Convertor()
    adapter.convert(cfg.source, cfg.output_dir, options)
    report = adapter.finalize_target()
    print(report)


if __name__ == "__main__":
    run_miku_hdf5_adapter()
