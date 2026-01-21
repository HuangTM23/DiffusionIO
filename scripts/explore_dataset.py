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
    top_dirs = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]
    print(f"Top-level directories ({len(top_dirs)}):")
    for d in sorted(top_dirs)[:10]:
        print(f"  - {d}")
    print()

    # 查找数据文件
    data_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith((".csv", ".h5", ".txt", ".json")):
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
