# lerobot_convertor

`lerobot_convertor` 是一个可扩展的数据集转换器框架：

- 输入：任意来源（HDF5、RLDS、自定义格式）
- 过程：用户做“读取 + 映射 + feature 对齐”
- 输出：官方 `LeRobotDataset`

本项目目标是**让用户轻松二次开发**，而不是把源数据规则硬编码在框架里。

特点：

- 统一转换流程，实现Any Dataset Format 转换 Lerobot Datasets Format
- 迭代式读取数据集进行转换，避免一次性读入内存造成内存爆炸，实现轻量性
---

## 目录

- [lerobot\_convertor](#lerobot_convertor)
  - [目录](#目录)
  - [1. 安装与使用](#1-安装与使用)
  - [2. 架构与职责](#2-架构与职责)
    - [封装好的部分（你不需要改）](#封装好的部分你不需要改)
    - [需要用户实现的部分（核心）](#需要用户实现的部分核心)
  - [3. 运行逻辑（统一流程）](#3-运行逻辑统一流程)
  - [4. 参数配置规范](#4-参数配置规范)
    - [`ConversionOptions` 字段](#conversionoptions-字段)
  - [5. feature 对齐规则（最重要）](#5-feature-对齐规则最重要)
  - [6. 指令增强（VLA 任务）](#6-指令增强vla-任务)
    - [任务选择工具函数使用](#任务选择工具函数使用)
  - [7. 二次开发位置指南](#7-二次开发位置指南)
    - [A. 新增你自己的 HDF5 适配器（推荐）](#a-新增你自己的-hdf5-适配器推荐)
    - [B. 你应该写在哪里](#b-你应该写在哪里)
  - [8. Miku 案例解读](#8-miku-案例解读)
    - [已封装部分](#已封装部分)
    - [Miku 用户实现部分](#miku-用户实现部分)
  - [9. 数据结构检查工具](#9-数据结构检查工具)
    - [HDF5 结构检查](#hdf5-结构检查)
    - [RLDS 结构检查](#rlds-结构检查)
  - [10. 常见问题](#10-常见问题)

---

## 1. 安装与使用

在项目目录安装（开发模式）：

```bash
conda create -n lerobot_convertor python=3.8 -y
conda activate lerobot_convertor
cd lerobot_convertor/
pip install -e .
```

运行 miku 示例：

```bash
python scripts/miku_hdf5_adapter.py \
  --source <your original dataset path> \
  --output_dir <lerobot datasets output path>
```

启用指令增强：

```bash
python scripts/miku_hdf5_adapter.py \
  --source <your original dataset path> \
  --output_dir <lerobot datasets output path> \
  --augment_task_instruction tr
```
此时需要在`--source <your original dataset path>`目录下创建`tasks_instruction.json`文件。

---

## 2. 架构与职责

### 封装好的部分（你不需要改）

- `LeRobotDatasetConvertor`（`lerobot_target.py`）
  - 调用官方 API 创建目标数据集
  - 自动遍历 episode / frame
  - 调用 `add_frame`、`save_episode`、`finalize`
  - 校验 `feature_values` key 是否与 `features` 对齐

- `Hdf5ToLeRobotConvertor` / `RldsToLeRobotConvertor`
  - 提供薄模板适配层
  - 暴露用户应重写的入口

### 需要用户实现的部分（核心）

- 确定好你的`feature`字典（即meta/info.json的feature部分），将其传入`DatasetsConvertorConfig.features`，并重写`extract_episode_from_file(...)`函数，实现从你的数据集文件里读取完整的一条episode数据，并将episode分割成每一frame。

用户必须保证每一frame结构为：

```python
{
  "task": "optional task string", 
  "timestamp": 0.0,  # optional
  "feature_values": { # 这里是你定义的feature
    "action": ...,
    "observation.state": ...,
    # 其他与 options.features 一致的 key
  },
}
```
以上数据均可由你自行决定。

---

## 3. 运行逻辑（统一流程）

`convert()` 的固定执行路径：

1. `iter_source_episodes(...)`：按读取顺序遍历源 episode
2. `build_episode(...)`：组装标准 episode
3. `build_frame(...)`：组装标准 frame
4. `add_frame(...)`：写入每帧
5. `save_episode()`：完成一条 episode
6. `finalize_target()`：写入 meta/stat 等收尾

---

## 4. 参数配置规范

配置定义统一在 `models.py`：

- `ConversionOptions`：转换运行参数
- `DatasetsConvertorConfig`：脚本级配置模板（可继承）

### `ConversionOptions` 字段

- `dataset_name: str` 输出数据集名称
- `fps: int` 视频/时间步频率
- `robot_type: str | None` 机器人类型
- `use_videos: bool` 是否启用视频特征写入
- `features: dict | None` 目标特征 schema（必须与 frame 对齐）
- `default_task: str` 默认任务文案
- `augment_task_instruction: bool` 是否启用指令增强（自行实现，非必要）
- `task_key: str` frame任务字段名（默认 `task`，不允许修改，否则无法通过Lerobot内部检查）
- `timestamp_key: str` 时间字段名（默认 `timestamp`）
---

## 5. feature 对齐规则（最重要）

`options.features` 中每个 key，必须在每帧 `feature_values` 中出现。

如果 `features` 是：

```python
{
  "observation.images.image": {...},
  "observation.state": {...},
  "action": {...},
}
```

则每帧必须包含返回：

```python
{
  "feature_values": {
    "observation.images.image": ...,
    "observation.state": ...,
    "action": ...,
  }
}
```

否则会在 `build_frame` / `add_frame` 阶段报错。

---

## 6. 指令增强（VLA 任务）

当 `augment_task_instruction=True`：

- 从 `cfg.source/tasks_instruction.json` 读取指令池
- 每条 episode 随机采样一条 instruction 作为该 episode 的 `task`

当 `augment_task_instruction=False`：

- 使用 `default_task`

支持的 `tasks_instruction.json` 格式：

```json
[
  "pick the banana",
  "grasp banana and place into bowl"
]
```

或

```json
{
  "tasks": ["...", "..."]
}
```

或

```json
{
  "task_instructions": ["...", "..."]
}
```

### 任务选择工具函数使用

推荐在自定义 Adapter 中复用 `utils.py` 里的两个函数：

- `load_task_instructions(source_root)`：读取并解析 `tasks_instruction.json`
- `select_task_for_episode(options, source_root, instructions)`：按配置返回 episode task

```python
from lerobot_convertor.utils import load_task_instructions, select_task_for_episode

class YourHdf5Convertor(Hdf5ToLeRobotConvertor):
  def __init__(self) -> None:
    super().__init__()
    self._source_root = None
    self._task_instructions_cache = None

  def iter_source_episodes(self, source, options):
    source_path = Path(source)
    self._source_root = source_path if source_path.is_dir() else source_path.parent
    self._task_instructions_cache = None
    if options.augment_task_instruction:
      self._task_instructions_cache = load_task_instructions(self._source_root)
    return super().iter_source_episodes(source, options)

  def extract_episode_from_file(self, file_obj, file_path, episode_id, options):
    task = select_task_for_episode(
      options=options,
      source_root=self._source_root,
      instructions=self._task_instructions_cache,
    )
    ...
```

工具行为说明：

- 当 `augment_task_instruction=False` 时，直接返回 `default_task`
- 当 `augment_task_instruction=True` 时，优先使用传入的 `instructions`，否则从 `<source>/tasks_instruction.json` 加载并随机选择任务文案
- 若启用增强但文件不存在、为空或无有效字符串，会抛出明确异常

---

## 7. 二次开发位置指南

### A. 新增你自己的 HDF5 适配器（推荐）

1. 继承 `Hdf5ToLeRobotConvertor`
2. 实现 `extract_episode_from_file(...)`
3. 在脚本函数里配置 `features`
4. 调用 `convert(...)` + `finalize_target()`

### B. 你应该写在哪里

- 业务映射逻辑：`lc_scripts/your_adapter.py` 的 `extract_episode_from_file`
- 参数默认值：你自己的 `Config` dataclass（继承 `DatasetsConvertorConfig`）
- 特征定义：脚本 `run_xxx_adapter` 中的 `feature` 字典

---

## 8. Miku 案例解读

参考文件：`lc_scripts/miku_hdf5_adapter.py`

### 已封装部分

- HDF5 文件遍历
- episode/frame 标准化
- LeRobot 写入与收尾

### Miku 用户实现部分

- 从以下字段读取：
  - `cam_head/color`
  - `cam_wrist/color`
  - `left_arm/joint`
  - `left_arm/qpos`
  - `left_arm/gripper`
- 在 `extract_episode_from_file` 中构造：
  - `observation.images.image`（HWC->CHW）
  - `observation.images.wrist_image`（HWC->CHW）
  - `observation.state`
  - `action`
- 可选启用 `augment_task_instruction`

---

## 9. 数据结构检查工具

在编写适配器（Adapter）之前，建议先检查原始数据的结构，以确定 `features` 映射方式。

### HDF5 结构检查

可以使用提供的脚本快速打印 HDF5 树状结构、Dataset 形状及属性：

```bash
python scripts/inspect_hdf5.py <your_file.hdf5>
```

或在代码中调用：

```python
from lerobot_convertor.utils import print_hdf5_structure

print_hdf5_structure(source="path/to/data.hdf5", max_depth=10)
```

### RLDS 结构检查

对于 RLDS 格式，可以对迭代器采样打印结构：

```python
from lerobot_convertor.utils import print_rlds_structure

# source 参数为 RLDS 的数据集迭代对象
print_rlds_structure(source=dataset_iterable, max_episodes=3)
```

---

## 10. 常见问题

- **Q: 为什么 frame 校验报 key 缺失？**
  - A: `feature_values` 没覆盖 `features` 的全部 key。

- **Q: 为什么有编码日志（SVT）？**
  - A: 视觉特征 `dtype=video` 时会触发编码流程，属正常行为。

- **Q: 可以自动覆写已有输出目录吗？**
  - A: 不建议，框架默认不提供自动覆写，请手动清理旧目录。
