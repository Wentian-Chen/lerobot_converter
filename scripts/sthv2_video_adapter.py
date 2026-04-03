"""
uv run scripts/sthv2_video_adapter.py \
    --source D:/gitdownload/datasets/sthv2 \
    --output_dir D:/gitdownload/datasets/sthv2_lerobot_padding \
    --dataset_name sthv2_lerobot_padding \
    --image_mode center_crop \
    --target_height 224 \
    --target_width 224 \
    --robot_type HumanVideo \
    --use_videos True \
    --fps 12 \
    --max_episodes 500
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import io
import json
import sys
import tarfile
from typing import Any
import zipfile

import draccus
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from lerobot_converter.lerobot_target import LeRobotDatasetConverter
from lerobot_converter.models import ConversionOptions, DatasetsConverterConfig


class Sthv2VideoConverter(LeRobotDatasetConverter[str | Path]):
    """Convert Something-Something-v2 videos into LeRobot episodes.

    One video corresponds to one episode, and each decoded frame is one step.
    The language instruction is written into frame task text.
    """

    VIDEO_EXTS = {".webm", ".mp4", ".avi", ".mov", ".mkv"}
    VALID_IMAGE_MODES = {"resize_pad", "center_crop"}

    def __init__(
        self,
        split: str = "all",
        task_text_source: str = "label",
        image_mode: str = "resize_pad",
        target_height: int | None = None,
        target_width: int | None = None,
        max_episodes: int | None = None,
    ) -> None:
        super().__init__()
        self.split = self._normalize_split(split)
        self.task_text_source = task_text_source
        self.image_mode = image_mode.lower()
        if self.image_mode not in self.VALID_IMAGE_MODES:
            raise ValueError(
                f"image_mode must be one of {sorted(self.VALID_IMAGE_MODES)}, got: {image_mode}"
            )

        self.target_height = target_height
        self.target_width = target_width

        if max_episodes is not None and max_episodes <= 0:
            self.max_episodes = None
        else:
            self.max_episodes = max_episodes

    def iter_source_episodes(
        self,
        source: str | Path,
        options: ConversionOptions,
    ):
        source_dir = Path(source)
        labels_zip_path = self._resolve_labels_zip_path(source_dir)
        videos_tar_path = source_dir / "20bn-something-something-v2.tar.gz"

        if not videos_tar_path.exists():
            raise FileNotFoundError(f"20bn-something-something-v2.tar.gz not found: {videos_tar_path}")

        id_to_text = self._load_instruction_map(
            labels_zip_path,
            self.split,
            task_text_source=self.task_text_source,
        )
        if not id_to_text:
            raise ValueError("No annotations found in labels.zip for requested split.")

        def _episodes():
            episode_id = 0
            if (self.target_height is None) != (self.target_width is None):
                raise ValueError("target_height and target_width must be both set or both unset.")

            if self.target_height is not None and self.target_width is not None:
                target_height, target_width = int(self.target_height), int(self.target_width)
            else:
                target_height, target_width = self._get_target_image_shape(options)

            if target_height <= 0 or target_width <= 0:
                raise ValueError("Target image height/width must be positive integers.")

            state_padding_vector = self._make_padding_vector(
                options,
                feature_key="observation.state",
                default_size=7,
            )
            action_padding_vector = self._make_padding_vector(
                options,
                feature_key="action",
                default_size=7,
            )

            converted = 0
            skipped = 0
            with tarfile.open(videos_tar_path, mode="r:gz") as tar_obj:
                candidate_members = []
                for member in tar_obj:
                    if not member.isfile():
                        continue

                    suffix = Path(member.name).suffix.lower()
                    if suffix not in self.VIDEO_EXTS:
                        continue

                    video_id = Path(member.name).stem
                    if video_id not in id_to_text:
                        continue

                    candidate_members.append(member)

                available_count = len(candidate_members)
                if self.max_episodes is not None:
                    candidate_members = candidate_members[: self.max_episodes]

                selected_count = len(candidate_members)
                print(
                    "[sthv2] split="
                    f"{self.split}, annotations={len(id_to_text)}, available={available_count}, "
                    f"selected={selected_count}, image_mode={self.image_mode}, "
                    f"target={target_width}x{target_height}"
                )

                if selected_count == 0:
                    raise ValueError("No videos matched current split/filter settings.")

                for member in candidate_members:
                    video_id = Path(member.name).stem

                    extracted = tar_obj.extractfile(member)
                    if extracted is None:
                        skipped += 1
                        continue

                    video_bytes = extracted.read()
                    frames = self._decode_video_bytes(video_bytes)
                    if not frames:
                        skipped += 1
                        continue

                    task_text = id_to_text.get(video_id, options.default_task)
                    steps = []
                    for idx, frame_hwc in enumerate(frames):
                        processed = self._process_frame(
                            frame_hwc,
                            target_height=target_height,
                            target_width=target_width,
                        )
                        chw = np.transpose(processed, (2, 0, 1))
                        steps.append(
                            {
                                "task": task_text,
                                "timestamp": float(idx / options.fps),
                                "feature_values": {
                                    "observation.images.image": chw,
                                    "observation.state": state_padding_vector.copy(),
                                    "action": action_padding_vector.copy(),
                                },
                            }
                        )

                    converted += 1
                    yield {
                        "episode_id": episode_id,
                        "steps": steps,
                    }
                    episode_id += 1

            print(
                f"[sthv2] conversion summary: converted={converted}, skipped={skipped}, requested={selected_count}"
            )

        return _episodes()

    def build_frame(self, source_frame: Any, options: ConversionOptions):
        if not isinstance(source_frame, dict):
            raise TypeError("Source frame must be a dictionary.")

        values = source_frame.get("feature_values")
        if not isinstance(values, dict):
            raise ValueError("Source frame must include dict field 'feature_values'.")

        required_keys = set((options.features or {}).keys())
        missing_keys = required_keys - set(values.keys())
        if missing_keys:
            raise ValueError(f"Frame missing required feature keys: {sorted(missing_keys)}")

        from lerobot_converter.models import NormalizedFrame

        return NormalizedFrame(
            task=str(source_frame.get("task", options.default_task)),
            feature_values=values,
            timestamp=source_frame.get("timestamp"),
        )

    def build_episode(self, source_episode: Any, episode_index: int, options: ConversionOptions):
        if not isinstance(source_episode, dict):
            raise TypeError(f"Episode {episode_index} must be a dictionary.")
        source_steps = source_episode.get("steps")
        if not isinstance(source_steps, list):
            raise ValueError(f"Episode {episode_index} has invalid steps structure.")

        from lerobot_converter.models import NormalizedEpisode

        frames = tuple(self.build_frame(source_step, options) for source_step in source_steps)
        return NormalizedEpisode(
            episode_id=int(source_episode.get("episode_id", episode_index)),
            frames=frames,
        )

    @staticmethod
    def _normalize_split(split: str) -> str:
        split_norm = split.lower().strip()
        if split_norm in {"valid", "val"}:
            return "validation"
        return split_norm

    @staticmethod
    def _resolve_labels_zip_path(source_dir: Path) -> Path:
        for zip_name in ("labels.zip", "label.zip"):
            zip_path = source_dir / zip_name
            if zip_path.exists():
                return zip_path
        raise FileNotFoundError(
            f"Neither labels.zip nor label.zip found under source directory: {source_dir}"
        )

    @staticmethod
    def _make_padding_vector(
        options: ConversionOptions,
        feature_key: str,
        default_size: int = 7,
    ) -> np.ndarray:
        feature = (options.features or {}).get(feature_key)
        if isinstance(feature, dict):
            shape = feature.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) == 1:
                size = int(shape[0])
                if size > 0:
                    return np.zeros((size,), dtype=np.float64)
        return np.zeros((default_size,), dtype=np.float64)

    @classmethod
    def _load_instruction_map(
        cls,
        labels_zip_path: Path,
        split: str,
        task_text_source: str = "label",
    ) -> dict[str, str]:
        split = cls._normalize_split(split)
        valid_splits = {"train", "validation", "test", "all"}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {sorted(valid_splits)}, got: {split}")

        valid_sources = {"label", "template"}
        source = task_text_source.lower()
        if source not in valid_sources:
            raise ValueError(
                f"task_text_source must be one of {sorted(valid_sources)}, got: {task_text_source}"
            )

        id_to_text: dict[str, str] = {}

        with zipfile.ZipFile(labels_zip_path, mode="r") as zip_obj:
            zip_names = set(zip_obj.namelist())

            def _load_json(name: str) -> list[dict[str, Any]]:
                with zip_obj.open(name, mode="r") as fobj:
                    return json.load(io.TextIOWrapper(fobj, encoding="utf-8"))

            def _pick_existing_name(*candidates: str) -> str | None:
                for candidate in candidates:
                    if candidate in zip_names:
                        return candidate
                return None

            split_files: list[tuple[str, ...]] = []
            if split in {"train", "all"}:
                split_files.append(("labels/train.json", "train.json"))
            if split in {"validation", "all"}:
                split_files.append(("labels/validation.json", "validation.json"))
            if split in {"test", "all"}:
                split_files.append(("labels/test.json", "test.json"))

            for split_file_candidates in split_files:
                split_file = _pick_existing_name(*split_file_candidates)
                if split_file is None:
                    continue
                for item in _load_json(split_file):
                    video_id = str(item["id"])
                    raw_label = str(item.get("label", "")).strip()
                    raw_template = str(item.get("template", "")).strip()

                    if source == "label":
                        text = raw_label or raw_template
                    else:
                        text = raw_template or raw_label

                    if text:
                        id_to_text[video_id] = text

            answers_name = _pick_existing_name("labels/test-answers.csv", "test-answers.csv")
            if answers_name is not None and split in {"test", "all"}:
                with zip_obj.open(answers_name, mode="r") as fobj:
                    reader = csv.reader(io.TextIOWrapper(fobj, encoding="utf-8"), delimiter=";")
                    for row in reader:
                        if len(row) < 2:
                            continue
                        video_id = row[0].strip()
                        answer_text = row[1].strip()
                        if video_id:
                            id_to_text[video_id] = answer_text

        return id_to_text

    @staticmethod
    def _decode_video_bytes(video_bytes: bytes) -> list[np.ndarray]:
        try:
            import av
        except ImportError as exc:
            raise ImportError("PyAV is required to decode videos. Please install 'av'.") from exc

        frames: list[np.ndarray] = []
        with av.open(io.BytesIO(video_bytes), mode="r") as container:
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="rgb24")
                frames.append(arr)
        return frames

    @staticmethod
    def _get_target_image_shape(options: ConversionOptions) -> tuple[int, int]:
        if options.features is None or "observation.images.image" not in options.features:
            raise ValueError("options.features must include 'observation.images.image'.")

        shape = options.features["observation.images.image"].get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            raise ValueError("'observation.images.image.shape' must be a 3D shape like (3, H, W).")

        channels, height, width = int(shape[0]), int(shape[1]), int(shape[2])
        if channels != 3:
            raise ValueError("'observation.images.image' currently requires 3 channels.")
        if height <= 0 or width <= 0:
            raise ValueError("Target image height/width must be positive integers.")
        return height, width

    @staticmethod
    def _resize_with_aspect_ratio_and_pad(
        frame_hwc: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        src_h, src_w = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
        if src_h <= 0 or src_w <= 0:
            raise ValueError("Invalid frame shape.")

        scale = min(target_width / src_w, target_height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        pil_img = Image.fromarray(frame_hwc)
        resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
        resized_arr = np.asarray(resized, dtype=np.uint8)

        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y0 = (target_height - new_h) // 2
        x0 = (target_width - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized_arr
        return canvas

    @staticmethod
    def _resize_then_center_crop(
        frame_hwc: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        src_h, src_w = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
        if src_h <= 0 or src_w <= 0:
            raise ValueError("Invalid frame shape.")

        scale = max(target_width / src_w, target_height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        pil_img = Image.fromarray(frame_hwc)
        resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
        resized_arr = np.asarray(resized, dtype=np.uint8)

        y0 = max(0, (new_h - target_height) // 2)
        x0 = max(0, (new_w - target_width) // 2)
        return resized_arr[y0 : y0 + target_height, x0 : x0 + target_width]

    def _process_frame(
        self,
        frame_hwc: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        if self.image_mode == "resize_pad":
            return self._resize_with_aspect_ratio_and_pad(frame_hwc, target_height, target_width)
        if self.image_mode == "center_crop":
            return self._resize_then_center_crop(frame_hwc, target_height, target_width)
        raise ValueError(f"Unsupported image_mode: {self.image_mode}")


def infer_first_frame_shape(source_dir: str | Path) -> tuple[int, int]:
    videos_tar_path = Path(source_dir) / "20bn-something-something-v2.tar.gz"
    if not videos_tar_path.exists():
        raise FileNotFoundError(f"20bn-something-something-v2.tar.gz not found: {videos_tar_path}")

    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV is required to decode videos. Please install 'av'.") from exc

    with tarfile.open(videos_tar_path, mode="r:gz") as tar_obj:
        for member in tar_obj:
            if not member.isfile():
                continue
            if Path(member.name).suffix.lower() not in Sthv2VideoConverter.VIDEO_EXTS:
                continue
            extracted = tar_obj.extractfile(member)
            if extracted is None:
                continue
            video_bytes = extracted.read()
            with av.open(io.BytesIO(video_bytes), mode="r") as container:
                for frame in container.decode(video=0):
                    arr = frame.to_ndarray(format="rgb24")
                    height, width = arr.shape[:2]
                    return int(height), int(width)
    raise ValueError("Could not decode any video frame to infer image shape.")


@dataclass
class Sthv2VideoAdapterConfig(DatasetsConverterConfig):
    source: str = ""
    output_dir: str = ""
    dataset_name: str = "sthv2_lerobot"
    fps: int = 12
    robot_type: str | None = ""
    use_videos: bool = True
    features: dict[str, dict[str, Any]] | None = None
    default_task: str = ""
    augment_task_instruction: bool = False
    # "train", "validation", "test", "all"
    split: str = "train"
    task_text_source: str = "label"
    # "center_crop" or "resize_pad"
    image_mode: str = "center_crop"
    target_height: int = 224
    target_width: int = 224
    # 0 or negative means no limit
    max_episodes: int = 0


@draccus.wrap()
def run_sthv2_video_adapter(cfg: Sthv2VideoAdapterConfig):
    if not cfg.source:
        raise ValueError("Provide --source path to directory containing labels.zip and video tar.")
    if not cfg.output_dir:
        raise ValueError("Provide --output_dir for converted dataset output.")

    options = cfg.options

    if (cfg.target_height > 0) != (cfg.target_width > 0):
        raise ValueError("target_height and target_width must be both positive or both non-positive.")

    if cfg.target_height > 0 and cfg.target_width > 0:
        target_height, target_width = int(cfg.target_height), int(cfg.target_width)
    elif options.features is not None and "observation.images.image" in options.features:
        target_height, target_width = Sthv2VideoConverter._get_target_image_shape(options)
    else:
        target_height, target_width = infer_first_frame_shape(cfg.source)

    feature = dict(options.features or {})
    
    # Configure image feature based on use_videos flag
    if options.use_videos:
        feature["observation.images.image"] = {
            "dtype": "video",
            "shape": (3, target_height, target_width),
            "names": ["channels", "height", "width"],
            "info": {
                "video.height": target_height,
                "video.width": target_width,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": options.fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    else:
        # Use image dtype when not using videos
        feature["observation.images.image"] = {
            "dtype": "uint8",
            "shape": (3, target_height, target_width),
            "names": ["channels", "height", "width"],
        }
    
    feature["observation.state"] = {
        "dtype": "float64",
        "shape": (7,),
        "names": [
            "state_padding_1",
            "state_padding_2",
            "state_padding_3",
            "state_padding_4",
            "state_padding_5",
            "state_padding_6",
            "state_padding_7",
        ],
    }
    feature["action"] = {
        "dtype": "float64",
        "shape": (7,),
        "names": [
            "action_padding_1",
            "action_padding_2",
            "action_padding_3",
            "action_padding_4",
            "action_padding_5",
            "action_padding_6",
            "action_padding_7",
        ],
    }

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

    adapter = Sthv2VideoConverter(
        split=cfg.split,
        task_text_source=cfg.task_text_source,
        image_mode=cfg.image_mode,
        target_height=target_height,
        target_width=target_width,
        max_episodes=cfg.max_episodes,
    )
    adapter.convert(cfg.source, cfg.output_dir, options)
    report = adapter.finalize_target()
    print(report)


if __name__ == "__main__":
    run_sthv2_video_adapter()  # type: ignore[call-arg]
