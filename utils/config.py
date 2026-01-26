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

    # 运行输出
    output_dir: str = "experiments/runs"
    max_steps: Optional[int] = None


@dataclass
class DataConfig:
    """数据配置"""

    data_root: str = "/home/dawn/datas/IO_Datasets/FRDR"
    ronin_lists_root: str = "external/ronin/lists"

    # 数据预处理
    window_size: int = 200
    # NOTE: historical name kept for backward compatibility.
    stride: int = 100
    # Sampling step size for dataset indexing.
    step_size: int = 10
    # RoNIN velocity interval (w): v[t] = (pos[t+w]-pos[t])/(ts[t+w]-ts[t]).
    # For ~1s average at 200Hz, set to 200.
    velocity_interval: int = 1
    normalize: bool = True

    # Dataset type: "strided" returns targ (B,2); "seq2seq" returns targ (B,2,T).
    dataset_type: str = "strided"

    # Optional explicit list files (preferred).
    train_list: Optional[str] = None
    val_list: Optional[str] = None
    test_list: Optional[str] = None

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

    # Cascade condition settings
    use_ronin_condition: bool = True
    condition_channels: int = 8

    # UNet1D core hyperparams
    unet_base_channels: int = 64
    unet_channel_mults: list = field(default_factory=lambda: [1, 2, 4, 8])
    unet_time_emb_dim: int = 128


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
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # 递归转换字典为配置对象
    if "training" in config_dict:
        config_dict["training"] = TrainingConfig(**config_dict["training"])
    if "data" in config_dict:
        config_dict["data"] = DataConfig(**config_dict["data"])
    if "model" in config_dict:
        config_dict["model"] = ModelConfig(**config_dict["model"])

    return ExperimentConfig(**config_dict)


def save_config_to_yaml(config: ExperimentConfig, yaml_path: str):
    """保存配置到YAML文件"""
    config_dict = {
        "name": config.name,
        "description": config.description,
        "seed": config.seed,
        "training": config.training.__dict__,
        "data": config.data.__dict__,
        "model": config.model.__dict__,
    }

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> ExperimentConfig:
    """获取默认配置"""
    return ExperimentConfig()
