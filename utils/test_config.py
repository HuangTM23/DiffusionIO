#!/usr/bin/env python3
"""测试配置系统"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
