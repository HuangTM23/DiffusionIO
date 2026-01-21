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