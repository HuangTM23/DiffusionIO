"""
日志工具
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import wandb


def setup_logger(
    name: str = "DiffusionIO", log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
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
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class WandBLogger:
    """WandB日志记录器"""

    def __init__(
        self,
        project: str,
        config: dict,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        enabled: bool = True,
    ):
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
                settings=wandb.Settings(start_method="thread"),
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
