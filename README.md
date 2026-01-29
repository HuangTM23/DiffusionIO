# DiffusionIO: IMU-based Velocity Estimation with Diffusion Models

本项目使用扩散模型进行 2D 速度估计，支持基线对比、级联优化和端到端生成三种模式。

## 🚀 技术方案

| 模式 | 名称 | 核心架构 | 配置脚本 |
| :--- | :--- | :--- | :--- |
| **Baseline** | **RoNIN** | ResNet18 (Regression) | `ronin_resnet.yaml` |
| **Scheme 1** | **Cascade** | RoNIN + 1D-UNet | `phase2_cascade.yaml` |
| **Scheme 2** | **End-to-End** | Pure 1D-UNet | `phase3_end2end.yaml` |

## 🛠️ 快速开始

### 1. 环境与数据准备
```bash
# 克隆并创建环境
git clone --recursive https://github.com/HuangTM23/DiffusionIO.git
conda env create -f environment.yml && conda activate DiffM

# 准备数据集 (将 ZIP 包放至 data/datasets/FRDR/Data/)
python scripts/prepare_frdr_extracted.py --delete-zips
```

### 2. 全局 WandB 配置
编辑 `experiments/configs/wandb.yaml` 统一管理实验记录：
```yaml
project: "DiffusionIO"
entity: "your-username"
mode: "online"  # 可选: online, offline, disabled
```

## 🏃‍♂️ 运行指南 (推荐使用一键配置模式)

所有脚本均支持 `--config` 参数，命令行会自动加载 YAML 中的所有设置。

### 0. RoNIN 基线 (Baseline)
*   **训练**: `python scripts/train_ronin_resnet.py train --config experiments/configs/ronin_resnet.yaml`
*   **测试**: `python scripts/train_ronin_resnet.py test --config experiments/configs/ronin_resnet.yaml`

### 1. 方案一 (Cascade)
*   **注意**: 需先在 `phase2_cascade.yaml` 中指定 `ronin_model_path` 为基线权重路径。
*   **训练**: `python scripts/train_phase2_cascade.py --config experiments/configs/phase2_cascade.yaml`
*   **评估**: `python scripts/eval_phase2_cascade.py --config experiments/configs/phase2_cascade.yaml`

### 2. 方案二 (End-to-End)
*   **训练**: `python scripts/train_phase3_end2end.py --config experiments/configs/phase3_end2end.yaml`
*   **评估**: `python scripts/eval_phase3_end2end.py --config experiments/configs/phase3_end2end.yaml`

## 📂 项目结构
```
DiffusionIO/
├── data/loaders/       # 数据加载适配器
├── models/             # 模型架构 (RoNIN, UNet, Hybrid)
├── training/           # 训练核心逻辑
├── evaluation/         # 评估核心逻辑
├── experiments/configs/# 全套 YAML 配置文件
└── scripts/            # CLI 运行入口
```