# Phase 1: 环境设置和数据准备实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 设置conda DiffM环境，克隆Ronin源代码，解压并验证FRDR数据集的数据加载流程

**Architecture:** 基于Ronin源代码的数据管道，使用其现有的数据加载器和预处理脚本

**Tech Stack:** Conda, PyTorch, Git, Ronin源代码

---

## Task 1: 创建项目基础目录结构

**Files:**
- Create: `data/loaders/`
- Create: `data/preprocessing/`
- Create: `data/datasets/`
- Create: `models/ronin/`
- Create: `models/diffusion/`
- Create: `models/hybrid/`
- Create: `training/trainers/`
- Create: `training/losses/`
- Create: `training/schedulers/`
- Create: `evaluation/metrics/`
- Create: `evaluation/visualization/`
- Create: `evaluation/benchmarks/`
- Create: `experiments/configs/`
- Create: `experiments/runs/`
- Create: `scripts/`
- Create: `utils/`
- Create: `docs/plans/`

**Step 1: 创建所有目录**

Run:
```bash
mkdir -p data/loaders data/preprocessing data/datasets
mkdir -p models/ronin models/diffusion models/hybrid
mkdir -p training/trainers training/losses training/schedulers
mkdir -p evaluation/metrics evaluation/visualization evaluation/benchmarks
mkdir -p experiments/configs experiments/runs
mkdir -p scripts utils docs/plans
```

Expected: 所有目录创建成功，无错误

**Step 2: 验证目录结构**

Run:
```bash
tree -L 2 -d
```

Expected: 显示完整的目录结构，包括所有创建的目录

**Step 3: 初始化Git仓库**

Run:
```bash
git init
```

Expected: Initialized empty Git repository

**Step 4: 创建.gitignore**

Create: `.gitignore`
```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data
data/raw/
data/processed/
*.zip
*.tar.gz

# Experiments
experiments/runs/
wandb/

# Model checkpoints
checkpoints/
*.pth
*.pt

# Logs
logs/
*.log
EOF
```

**Step 5: 提交初始结构**

Run:
```bash
git add .gitignore
git commit -m "feat: initialize project structure and .gitignore"
```

Expected: 成功提交

---

## Task 2: 创建Conda环境配置文件

**Files:**
- Create: `environment.yml`

**Step 1: 创建environment.yml**

Create: `environment.yml`
```yaml
name: DiffM
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - torchaudio>=2.0.0
  - pytorch-cuda=11.8
  - numpy>=1.24.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - scipy>=1.10.0
  - scikit-learn>=1.2.0
  - tqdm>=4.65.0
  - tensorboard>=2.13.0
  - pip
  - pip:
    - wandb>=0.15.0
    - tensorboardX>=2.6.2
```

**Step 2: 创建requirements.txt**

Create: `requirements.txt`
```bash
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
scikit-learn>=1.2.0
tqdm>=4.65.0
tensorboard>=2.13.0
wandb>=0.15.0
tensorboardX>=2.6.2
jupyter>=1.0.0
ipython>=8.12.0
EOF
```

**Step 3: 提交配置文件**

Run:
```bash
git add environment.yml requirements.txt
git commit -m "feat: add conda and pip requirements"
```

---

## Task 3: 克隆Ronin源代码仓库

**Files:**
- Create: `external/ronin/` (external dependencies目录)

**Step 1: 创建external目录**

Run:
```bash
mkdir -p external
```

**Step 2: 克隆Ronin仓库**

Run:
```bash
cd external
git clone https://github.com/Sachini/ronin.git
cd ..
```

Expected: 克隆成功，external/ronin/目录存在

**Step 3: 探索Ronin代码结构**

Run:
```bash
ls -la external/ronin/
find external/ronin/ -name "*.py" -type f | head -20
```

Expected: 看到Python文件结构，包括数据加载器、模型定义等

**Step 4: 查找数据加载相关文件**

Run:
```bash
find external/ronin/ -name "*data*" -o -name "*dataset*" | grep -E "\.py$"
```

Expected: 找到数据加载相关的Python文件

**Step 5: 查找数据集lists**

Run:
```bash
find external/ronin/ -name "*list*" -o -name "*split*" | head -10
```

Expected: 找到训练/验证/测试集划分的文件

**Step 6: 添加Ronin到.gitignore**

Run:
```bash
echo "external/ronin/" >> .gitignore
git add .gitignore
git commit -m "chore: add ronin repo to gitignore"
```

---

## Task 4: 探索FRDR数据集

**Files:**
- None (只读操作)

**Step 1: 检查数据集文件**

Run:
```bash
ls -lh /home/dawn/datas/IO_Datasets/FRDR_dataset_538_download_758_202510091948.zip
```

Expected: 看到文件大小约14.8GB

**Step 2: 创建数据目录**

Run:
```bash
mkdir -p /home/dawn/datas/IO_Datasets/FRDR
```

**Step 3: 解压数据集（这可能需要很长时间）**

Run:
```bash
cd /home/dawn/datas/IO_Datasets/FRDR
unzip /home/dawn/datas/IO_Datasets/FRDR_dataset_538_download_758_202510091948.zip
```

Expected: 解压到FRDR目录，包含多个数据文件

**注意：** 这个步骤可能需要30分钟到1小时，取决于磁盘速度。可以考虑使用nohup或screen在后台运行。

**Step 4: 探索解压后的数据结构**

Run:
```bash
ls -la /home/dawn/datas/IO_Datasets/FRDR/ | head -20
```

Expected: 看到数据集的结构，可能包含多个序列文件夹

**Step 5: 查找示例数据文件**

Run:
```bash
find /home/dawn/datas/IO_Datasets/FRDR/ -name "*.csv" -o -name "*.h5" -o -name "*.txt" | head -10
```

Expected: 找到数据文件格式

**Step 6: 创建数据探索脚本**

Create: `scripts/explore_dataset.py`
```python
#!/usr/bin/env python3
"""
探索FRDR数据集结构
"""
import os
import sys

def explore_dataset(root_dir):
    """探索数据集目录结构"""
    print(f"Exploring dataset at: {root_dir}\n")

    # 列出顶层目录
    top_dirs = [d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Top-level directories ({len(top_dirs)}):")
    for d in sorted(top_dirs)[:10]:
        print(f"  - {d}")
    print()

    # 查找数据文件
    data_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(('.csv', '.h5', '.txt', '.json')):
                data_files.append(os.path.join(root, f))

    print(f"Data files found: {len(data_files)}")
    if data_files:
        print("Sample files:")
        for f in data_files[:5]:
            print(f"  - {f}")
    print()

    return data_files

if __name__ == "__main__":
    dataset_path = "/home/dawn/datas/IO_Datasets/FRDR"
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        sys.exit(1)

    explore_dataset(dataset_path)
```

**Step 7: 运行探索脚本**

Run:
```bash
python scripts/explore_dataset.py
```

Expected: 显示数据集的文件结构信息

**Step 8: 提交探索脚本**

Run:
```bash
git add scripts/explore_dataset.py
git commit -m "feat: add dataset exploration script"
```

---

## Task 5: 分析Ronin数据加载代码

**Files:**
- Read: `external/ronin/` (多个文件)

**Step 1: 查找数据加载器**

Run:
```bash
grep -r "class.*Dataset\|class.*Loader" external/ronin/*.py --include="*.py"
```

Expected: 找到数据集和加载器类的定义

**Step 2: 查找主入口文件**

Run:
```bash
ls external/ronin/*.py | head -10
```

Expected: 看到主要的Python文件

**Step 3: 读取README或文档**

Run:
```bash
cat external/ronin/README.md 2>/dev/null | head -100
```

Expected: 了解Ronin项目的基本用法和数据格式

**Step 4: 创建数据管道分析文档**

Create: `docs/ronin-data-pipeline-analysis.md`
```markdown
# Ronin数据管道分析

## 数据加载器位置
- [待填写]

## 数据格式
- [待填写]

## 预处理步骤
- [待填写]

## 数据集划分
- [待填写]

## 需要复制的文件
- [待填写]
```

**Step 5: 提交分析文档**

Run:
```bash
git add docs/ronin-data-pipeline-analysis.md
git commit -m "docs: add ronin data pipeline analysis template"
```

---

## Task 6: 创建基础配置系统

**Files:**
- Create: `utils/config.py`
- Create: `experiments/configs/base_config.yaml`

**Step 1: 创建配置工具模块**

Create: `utils/__init__.py`
```python
# Utils package
```

Create: `utils/config.py`
```python
"""
配置管理工具
"""
import yaml
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    # 保存和日志
    checkpoint_dir: str = "experiments/checkpoints"
    log_interval: int = 10
    save_interval: int = 5
    use_wandb: bool = True

    # WandB配置
    wandb_project: str = "DiffusionIO"
    wandb_entity: Optional[str] = None


@dataclass
class DataConfig:
    """数据配置"""
    data_root: str = "/home/dawn/datas/IO_Datasets/FRDR"
    ronin_lists_root: str = "external/ronin/lists"

    # 数据预处理
    window_size: int = 200
    stride: int = 100
    normalize: bool = True

    # 数据集划分
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


@dataclass
class ModelConfig:
    """模型配置"""
    # 输入输出维度
    imu_channels: int = 6  # 加速度计3 + 陀螺仪3
    velocity_channels: int = 2  # vx, vy

    # Ronin模型配置
    ronin_model_path: Optional[str] = None

    # 扩散模型配置
    diffusion_timesteps: int = 1000
    noise_schedule: str = "linear"  # linear, cosine

    # UNet配置
    unet_channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    attention_resolutions: list = field(default_factory=lambda: [4, 8])


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "baseline"
    description: str = ""
    seed: int = 42

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config_from_yaml(yaml_path: str) -> ExperimentConfig:
    """从YAML文件加载配置"""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 转换为配置对象（简化版，实际需要更详细的转换）
    return ExperimentConfig(**config_dict)


def save_config_to_yaml(config: ExperimentConfig, yaml_path: str):
    """保存配置到YAML文件"""
    config_dict = {
        'name': config.name,
        'description': config.description,
        'seed': config.seed,
        'training': config.training.__dict__,
        'data': config.data.__dict__,
        'model': config.model.__dict__,
    }

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig()
```

**Step 2: 创建基础配置文件**

Create: `experiments/configs/base_config.yaml`
```yaml
name: baseline
description: Baseline experiment configuration
seed: 42

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.00001
  gradient_clip: 1.0
  device: cuda
  num_workers: 4
  pin_memory: true
  checkpoint_dir: experiments/checkpoints
  log_interval: 10
  save_interval: 5
  use_wandb: true
  wandb_project: DiffusionIO
  wandb_entity: null

data:
  data_root: /home/dawn/datas/IO_Datasets/FRDR
  ronin_lists_root: external/ronin/lists
  window_size: 200
  stride: 100
  normalize: true
  train_split: train
  val_split: val
  test_split: test

model:
  imu_channels: 6
  velocity_channels: 2
  ronin_model_path: null
  diffusion_timesteps: 1000
  noise_schedule: linear
  unet_channels: [64, 128, 256, 512]
  attention_resolutions: [4, 8]
```

**Step 3: 测试配置系统**

Create: `utils/test_config.py`
```python
#!/usr/bin/env python3
"""测试配置系统"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config import get_default_config, load_config_from_yaml

def test_default_config():
    """测试默认配置"""
    config = get_default_config()
    print("Testing default config...")
    print(f"  Name: {config.name}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  IMU channels: {config.model.imu_channels}")
    print("  ✓ Default config test passed")

def test_yaml_config():
    """测试YAML配置加载"""
    config_path = "experiments/configs/base_config.yaml"
    if not os.path.exists(config_path):
        print(f"  ✗ Config file not found: {config_path}")
        return

    config = load_config_from_yaml(config_path)
    print("Testing YAML config loading...")
    print(f"  Name: {config.name}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Data root: {config.data.data_root}")
    print("  ✓ YAML config test passed")

if __name__ == "__main__":
    test_default_config()
    test_yaml_config()
    print("\nAll config tests passed!")
```

**Step 4: 运行配置测试**

Run:
```bash
python utils/test_config.py
```

Expected: 所有测试通过，显示配置信息

**Step 5: 提交配置系统**

Run:
```bash
git add utils/config.py utils/__init__.py utils/test_config.py experiments/configs/base_config.yaml
git commit -m "feat: add configuration system"
```

---

## Task 7: 创建日志和工具函数

**Files:**
- Create: `utils/logger.py`
- Create: `utils/__init__.py` (更新)

**Step 1: 创建日志工具**

Create: `utils/logger.py`
```python
"""
日志工具
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import wandb


def setup_logger(name: str = "DiffusionIO",
                 log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class WandBLogger:
    """WandB日志记录器"""

    def __init__(self, project: str,
                 config: dict,
                 name: Optional[str] = None,
                 entity: Optional[str] = None,
                 enabled: bool = True):
        """
        初始化WandB日志记录器

        Args:
            project: WandB项目名称
            config: 实验配置字典
            name: 实验名称
            entity: WandB实体名称
            enabled: 是否启用WandB
        """
        self.enabled = enabled

        if self.enabled:
            wandb.init(
                project=project,
                name=name,
                entity=entity,
                config=config,
                settings=wandb.Settings(
                    start_method="thread"
                )
            )

    def log_metrics(self, metrics: dict, step: int):
        """记录指标"""
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str, type: str = "model"):
        """记录工件"""
        if self.enabled:
            artifact = wandb.Artifact(name, type=type)
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def finish(self):
        """结束WandB会话"""
        if self.enabled:
            wandb.finish()
```

**Step 2: 创建通用工具函数**

Create: `utils/common.py`
```python
"""
通用工具函数
"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    设置随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型的可训练参数数量

    Args:
        model: PyTorch模型

    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    获取最佳可用设备

    Returns:
        torch.device对象
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_time(seconds: float) -> str:
    """
    格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
```

**Step 3: 更新__init__.py**

Update: `utils/__init__.py`
```python
"""
工具函数包
"""
from utils.config import (
    ExperimentConfig,
    TrainingConfig,
    DataConfig,
    ModelConfig,
    get_default_config,
    load_config_from_yaml,
    save_config_to_yaml
)
from utils.logger import setup_logger, WandBLogger
from utils.common import set_seed, count_parameters, get_device, format_time

__all__ = [
    # Config
    'ExperimentConfig',
    'TrainingConfig',
    'DataConfig',
    'ModelConfig',
    'get_default_config',
    'load_config_from_yaml',
    'save_config_to_yaml',
    # Logger
    'setup_logger',
    'WandBLogger',
    # Common
    'set_seed',
    'count_parameters',
    'get_device',
    'format_time',
]
```

**Step 4: 提交工具函数**

Run:
```bash
git add utils/logger.py utils/common.py utils/__init__.py
git commit -m "feat: add logger and common utilities"
```

---

## Task 8: 创建README和项目文档

**Files:**
- Create: `README.md`

**Step 1: 创建项目README**

Create: `README.md`
```markdown
# DiffusionIO - 基于扩散模型的IMU速度估计

## 项目概述

本项目基于6轴IMU数据，使用扩散模型进行2维速度估计（vx, vy），并与RoNIN基准模型进行对比研究。

## 技术方案

### 方案一：级联方法（RoNIN + 扩散模型）
1. 使用RoNIN ResNet模型作为预处理器，得到初始速度估计
2. 扩散模型以初始速度估计为条件进行去噪优化
3. 输出优化后的2维速度

### 方案二：端到端方法（直接扩散模型）
1. 直接以IMU数据为条件
2. 扩散模型直接学习从IMU数据到速度的映射
3. 输出直接估计的2维速度

## 环境设置

### 1. 创建Conda环境

```bash
conda env create -f environment.yml
conda activate DiffM
```

### 2. 安装依赖（可选）

```bash
pip install -r requirements.txt
```

## 项目结构

```
DiffusionIO/
├── data/                    # 数据相关
├── models/                  # 模型定义
├── training/                # 训练相关
├── evaluation/              # 评估相关
├── experiments/             # 实验配置
├── scripts/                 # 实用脚本
├── utils/                   # 工具函数
└── docs/                    # 文档
```

## 数据集

- **FRDR数据集**: /home/dawn/datas/IO_Datasets/FRDR_dataset_538_download_758_202510091948.zip
- **数据划分**: 使用Ronin源代码中的lists

## 使用方法

### 训练

```bash
python scripts/train.py --config experiments/configs/base_config.yaml
```

### 评估

```bash
python scripts/evaluate.py --checkpoint experiments/checkpoints/best_model.pth
```

### 可视化

```bash
python scripts/visualize.py --checkpoint experiments/checkpoints/best_model.pth
```

## 参考文献

- RoNIN: https://github.com/Sachini/ronin
- DDPM: Denoising Diffusion Probabilistic Models

## 许可证

MIT License
```

**Step 2: 提交README**

Run:
```bash
git add README.md
git commit -m "docs: add project README"
```

---

## Task 9: 验证环境

**Files:**
- None

**Step 1: 激活Conda环境**

Run:
```bash
conda activate DiffM
```

Expected: 环境激活成功

**Step 2: 检查PyTorch安装**

Run:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected: 显示PyTorch版本和CUDA状态

**Step 3: 检查WandB安装**

Run:
```bash
python -c "import wandb; print(f'WandB version: {wandb.__version__}')"
```

Expected: 显示WandB版本

**Step 4: 运行配置测试**

Run:
```bash
python utils/test_config.py
```

Expected: 所有测试通过

**Step 5: 创建环境验证脚本**

Create: `scripts/check_environment.py`
```python
#!/usr/bin/env python3
"""
环境检查脚本
"""
import sys
import os

def check_import(module_name, package_name=None):
    """检查模块是否可导入"""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} installed")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} not installed: {e}")
        return False

def main():
    """主检查函数"""
    print("Checking DiffusionIO environment...\n")

    # 检查Python版本
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    if sys.version_info < (3, 8):
        print("  ⚠ Warning: Python 3.8+ recommended")
    print()

    # 检查核心依赖
    checks = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
        ("wandb", "WandB"),
    ]

    results = []
    for module, package in checks:
        results.append(check_import(module, package))

    print()

    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")

    print()

    # 检查项目结构
    print("Checking project structure...")
    required_dirs = [
        "data", "models", "training", "evaluation",
        "experiments", "scripts", "utils", "docs"
    ]
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ missing")

    print()

    # 检查数据集
    dataset_path = "/home/dawn/datas/IO_Datasets/FRDR"
    if os.path.exists(dataset_path):
        print(f"✓ Dataset directory exists: {dataset_path}")
    else:
        print(f"⚠ Dataset directory not found: {dataset_path}")

    ronin_path = "external/ronin"
    if os.path.exists(ronin_path):
        print(f"✓ Ronin repo exists: {ronin_path}")
    else:
        print(f"⚠ Ronin repo not found: {ronin_path}")

    print()

    # 总结
    passed = sum(results)
    total = len(results)
    print(f"Environment check: {passed}/{total} dependencies installed")

    if passed == total:
        print("\n✓ All checks passed!")
        return 0
    else:
        print("\n✗ Some checks failed. Please install missing dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Step 6: 运行环境检查**

Run:
```bash
python scripts/check_environment.py
```

Expected: 显示所有依赖和项目结构的检查结果

**Step 7: 提交检查脚本**

Run:
```bash
git add scripts/check_environment.py
git commit -m "feat: add environment check script"
```

---

## 验收标准

完成本阶段后，应该能够：

1. ✓ 成功激活DiffM conda环境
2. ✓ 导入所有核心依赖（PyTorch, NumPy, Pandas, Matplotlib, WandB等）
3. ✓ 检测到CUDA（如果可用）
4. ✓ 成功加载配置文件
5. ✓ 初始化日志系统
6. ✓ 查看FRDR数据集的文件结构
7. ✓ 访问Ronin源代码

---

## 下一步

完成Phase 1后，进入**Phase 2: 基础模型实现**
- 复制并适配Ronin模型
- 实现基础扩散模型框架
- 实现1D UNet架构

---

*Phase 1完成标准：所有环境配置和数据管道准备就绪*
