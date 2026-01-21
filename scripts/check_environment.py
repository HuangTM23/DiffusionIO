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
        "data",
        "models",
        "training",
        "evaluation",
        "experiments",
        "scripts",
        "utils",
        "docs",
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
