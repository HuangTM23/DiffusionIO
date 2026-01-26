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

## 数据集准备

本项目使用FRDR数据集。数据集应以ZIP包形式放置在 `data/datasets/FRDR/Data/*.zip`。

使用以下命令自动解压并校验数据集：

```bash
python scripts/prepare_frdr_extracted.py --delete-zips
```

解压后的数据将位于 `data/datasets/FRDR/extracted`。

## 使用方法

### 1. 训练 RoNIN 基线模型 (Baseline)

在进行级联训练前，需要先训练一个RoNIN ResNet模型作为先验。

```bash
# 自动生成可用序列列表并开始训练
python scripts/train_ronin_resnet.py train \
  --data-root data/datasets/FRDR/extracted \
  --out-dir experiments/runs/ronin_resnet \
  --epochs 50 \
  --arch resnet18
```

训练完成后，权重将保存至 `experiments/runs/ronin_resnet/checkpoints/`。

### 2. 训练方案一：级联扩散模型 (Scheme 1 Cascade)

1. 更新配置文件 `experiments/configs/phase2_cascade.yaml`：
   - 将 `model.ronin_model_path` 指向你训练好的基线权重路径。
2. 运行训练：

```bash
python scripts/train_phase2_cascade.py --config experiments/configs/phase2_cascade.yaml
```

### 3. 评估 (Evaluation)

使用评估脚本计算指标（RMSE/MAE）并生成预测轨迹文件（.npz）：

```bash
python scripts/eval_phase2_cascade.py \
  --config experiments/configs/phase2_cascade.yaml \
  --ckpt experiments/runs/phase2_cascade_vavg/checkpoints/epoch_last.pt \
  --split val
```

评估产物将保存至 `experiments/runs/phase2_cascade_vavg/predictions/`。

## 参考文献

- RoNIN: https://github.com/Sachini/ronin
- DDPM: Denoising Diffusion Probabilistic Models

## 许可证

MIT License