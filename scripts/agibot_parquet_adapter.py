"""
Agibot Parquet to LeRobot v3.0 Converter

Environment setup:
    uv sync --python 3.10
    uv pip install -e ".[agibot]"

Example usage:
    # Read from extracted directory
    uv run scripts/agibot_parquet_adapter.py \
        --source ../clean_the_desktop_part_1 \
        --output_dir /path/to/lerobot_v3_output \
        --read_from_tar False

    # Read from compressed *.tar.gz.* split files
    uv run scripts/agibot_parquet_adapter.py \
        --source ../clean_the_desktop_part_1 \
        --output_dir /path/to/lerobot_v3_output \
        --read_from_tar True
"""

import json
import subprocess
import tempfile
from collections.abc import Iterable,Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import os
import draccus
import numpy as np
import pyarrow.parquet as pq

from lerobot_converter.hdf5_adapter import Hdf5ToLeRobotConverter
from lerobot_converter.models import ConversionOptions, DatasetsConverterConfig


class AgibotParquetConverter(Hdf5ToLeRobotConverter):
    """Adapter for Agibot datasets (LeRobot v2.1) to LeRobot v3.0

    Expected structure:
    - data/chunk-000/episode_XXXXXX.parquet
    - meta/episodes.jsonl, info.json
    - videos/chunk-000/observation.images.{cam_name}/episode_XXXXXX.mp4
    """

    def __init__(self):
        super().__init__()
        self._source_root: Path | None = None
        self._meta_info: Dict[str, Any] = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def _extract_tar_if_needed(self, source: Path, read_from_tar: bool) -> Path:
        if not read_from_tar:
            return source

        self._temp_dir = tempfile.TemporaryDirectory()
        extract_path = Path(self._temp_dir.name)
        print(f"Extracting .tar.gz files from {source} to {extract_path}...")

        # Find any matching tar.gz.* parts or regular tar.gz files
        has_extracted = False
        import subprocess

        for tar_file in source.glob("*.tar.gz*"):
            if tar_file.name.endswith(".000"):
                # Handle split archives like meta.tar.gz.000
                base_name = tar_file.name.replace(".000", ".*")
                cmd = f"cat {source}/{base_name} | tar xz -C {extract_path}"
                os.system(cmd)
                has_extracted = True
            elif tar_file.name.endswith(".tar.gz"):
                # Handle standard non-split tar files
                cmd = f"tar xzf {tar_file} -C {extract_path}"
                os.system(cmd)
                has_extracted = True

        if not has_extracted:
            print(f"Warning: No matching *.tar.gz* found in {source}")

        return extract_path

    def iter_source_episodes(self, source: str | Path, options: ConversionOptions) -> Generator[Dict[str, Any], None, None]:
        source_path = Path(source)
        
        # Load meta info.json
        info_path = source_path / "meta" / "info.json"
        with open(info_path, "r", encoding="utf-8") as f:
            self._meta_info = json.load(f)

        data_dir = source_path / "data"
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            for pq_file in sorted(chunk_dir.glob("episode_*.parquet")):
                episode_id_str = pq_file.stem.split("_")[-1]
                episode_id = int(episode_id_str)
                yield self.extract_episode_from_file(pq_file, episode_id, source_path, options)

    def extract_episode_from_file(self, pq_file: Path, episode_id: int, source_root: Path, options: ConversionOptions) -> Dict[str, Any]:
        """Read 159-dim state and 40-dim action, extract subset -> 21-dim"""
        table = pq.read_table(pq_file)
        df = table.to_pandas()

        # Extract mappings based on task spec
        # State: 1 (0) + 1 (1) + 14 (30-43) + 5 (75-79) -> total 21
        state_cols = (
            [0, 1] + 
            list(range(30, 44)) + 
            list(range(75, 80))
        )
        
        # Action: 1 (0) + 1 (1) + 14 (16-29) + 5 (33-37) -> total 21
        action_cols = (
            [0, 1] + 
            list(range(16, 30)) + 
            list(range(33, 38))
        )

        state_raw = np.stack(df["observation.state"].values)
        action_raw = np.stack(df["action"].values)
        
        state_new = state_raw[:, state_cols]
        action_new = action_raw[:, action_cols]
        timestamps = df["timestamp"].values

        # Build task instruction
        instruction_segments = self._meta_info.get("instruction_segments", {})
        high_level_instruction = self._meta_info.get("high_level_instruction", {})
        ep_high_level = high_level_instruction.get(str(episode_id), options.default_task)
        ep_segments = instruction_segments.get(str(episode_id), [])

        # Read videos
        # Ideally, LeRobotDataset.add_frame supports video paths or frames. In lerobot_converter we need to pass images if we are yielding steps.
        # Alternatively, if we preserve the original video WITHOUT reading frame-by-frame, we might directly copy videos.
        # However, to conform to standard build_frame pipeline, we typically load them. 
        import imageio
        cams = {"hand_left": [], "hand_right": [], "top_head": []}
        for cam_name in cams.keys():
            vid_path = source_root / "videos" / pq_file.parent.name / f"observation.images.{cam_name}" / f"episode_{episode_id:06d}.mp4"
            if vid_path.exists():
                reader = imageio.get_reader(vid_path)
                cams[cam_name] = [frame for frame in reader]
                reader.close()

        step_count = len(df)
        steps = []
        for idx in range(step_count):
            feature_values = {
                "observation.state": state_new[idx].astype(np.float32),
                "action": action_new[idx].astype(np.float32),
            }
            if len(cams["hand_left"]) > idx:
                feature_values["observation.images.hand_left"] = cams["hand_left"][idx]
            if len(cams["hand_right"]) > idx:
                feature_values["observation.images.hand_right"] = cams["hand_right"][idx]
            if len(cams["top_head"]) > idx:
                feature_values["observation.images.top_head"] = cams["top_head"][idx]

            steps.append({
                "task": ep_high_level,
                "timestamp": float(timestamps[idx]),
                "feature_values": feature_values
            })

        return {"episode_id": episode_id, "steps": steps}

    def convert(self, source: str | Path, output_dir: str | Path, options: ConversionOptions, read_from_tar: bool = False):
        try:
            source_path = self._extract_tar_if_needed(Path(source), read_from_tar)
            # Find the actual extracted root directory by looking for 'meta/info.json'
            if read_from_tar:
                found_info = list(source_path.rglob("meta/info.json"))
                if found_info:
                    source_path = found_info[0].parent.parent
                else:
                    raise FileNotFoundError(f"Could not find meta/info.json in extracted files under {source_path}")
            
            # Defer to parent convert method which handles dataset lifecycle (target initialization, build_episode, etc.)
            super().convert(source_path, output_dir, options)
        finally:
            if self._temp_dir:
                self._temp_dir.cleanup()


@dataclass
class AgibotAdapterConfig(DatasetsConverterConfig):
    source: str = ""
    output_dir: str = ""
    dataset_name: str = ""
    fps: int = 30
    robot_type: str | None = "agibot"
    use_videos: bool = True
    default_task: str = "Perform task"
    read_from_tar: bool = False

@draccus.wrap()
def run_agibot_adapter(cfg: AgibotAdapterConfig):
    if not cfg.source:
        raise ValueError("Provide --source path")
    if not cfg.output_dir:
        raise ValueError("Provide --output_dir")
    
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (21,),
            "names": ["left_effector", "right_effector"] + [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)] + [f"waist_{i}" for i in range(5)]
        },
        "action": {
            "dtype": "float32",
            "shape": (21,),
            "names": ["left_effector", "right_effector"] + [f"left_joint_{i}" for i in range(7)] + [f"right_joint_{i}" for i in range(7)] + [f"waist_{i}" for i in range(5)]
        },
        "observation.images.hand_left": {
            "dtype": "video",
            "video_info": {
                "video.is_depth_map": False,
                "video.fps": 30.0,
                "video.codec": "hevc",
                "video.pix_fmt": "yuv420p",
                "has_audio": False
            },
            "shape": [
                1056,
                1280,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
      ]
        },
        "observation.images.hand_right": {
            "dtype": "video",
            "video_info": {
                "video.is_depth_map": False,
                "video.fps": 30.0,
                "video.codec": "hevc",
                "video.pix_fmt": "yuv420p",
                "has_audio": False
            },
            "shape": [
                1056,
                1280,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
      ]
        },
        "observation.images.top_head": {
            "dtype": "video",
            "video_info": {
                "video.is_depth_map": False,
                "video.fps": 30.0,
                "video.codec": "hevc",
                "video.pix_fmt": "yuv420p",
                "has_audio": False
            },
            "shape": [
                400,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
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
            features=features,
            default_task=options.default_task,
        )

    adapter = AgibotParquetConverter()
    adapter.convert(cfg.source, cfg.output_dir, options, read_from_tar=cfg.read_from_tar)
    report = adapter.finalize_target()
    print(report)

if __name__ == "__main__":
    run_agibot_adapter()
