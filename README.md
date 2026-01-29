# DiffusionIO: IMU-based Velocity Estimation with Diffusion Models

本项目探索使用扩散模型（Diffusion Models）基于 6 轴 IMU 数据进行 2D 速度估计，并与 RoNIN 基准模型进行对比。

## 🚀 技术方案

| 方案 | 名称 | 架构 | 特点 |
| :--- | :--- | :--- | :--- |
| **Baseline** | **RoNIN** | ResNet18 | 传统的监督学习，作为先验和基准。 |
| **Scheme 1** | **Cascade** | RoNIN + Diffusion | **级联架构**。RoNIN 提供粗略速度先验，扩散模型进行细化生成。 |
| **Scheme 2** | **End-to-End** | Pure Diffusion | **端到端架构**。直接以 IMU 数据为条件，由扩散模型独立学习速度映射。 |

## 🛠️ 环境与数据

### 1. 环境配置

```bash
# 1. 克隆仓库 (包含子模块)
git clone --recursive https://github.com/HuangTM23/DiffusionIO.git
cd DiffusionIO

# 2. 创建环境
conda env create -f environment.yml
conda activate DiffM
```

### 2. 数据准备

将 FRDR 数据集 ZIP 包放置于 `data/datasets/FRDR/Data/`，然后运行：

```bash
python scripts/prepare_frdr_extracted.py --delete-zips
```

## ⚙️ 实验配置 (WandB)

本项目使用 **Weights & Biases (WandB)** 跟踪实验。建议在 `experiments/configs/` 下的 YAML 文件中预先配置：

```yaml
training:
  use_wandb: true
  wandb_project: "DiffusionIO"
  wandb_entity: "your-entity"  # 可选
```

## 🏃‍♂️ 训练与评估

### 步骤 0: 训练 RoNIN 基线 (必须)

方案一需要 RoNIN 权重作为先验。

```bash
python scripts/train_ronin_resnet.py train \
  --data-root data/datasets/FRDR/extracted \
  --out-dir experiments/runs/ronin_resnet \
  --epochs 50 \
  --wandb-project DiffusionIO
```

### 步骤 1: 运行方案一 (Cascade)

1. 修改 `experiments/configs/phase2_cascade.yaml`，将 `ronin_model_path` 指向步骤 0 得到的权重。
2. **训练**:
   ```bash
   python scripts/train_phase2_cascade.py --config experiments/configs/phase2_cascade.yaml
   ```
3. **评估**:
   ```bash
   python scripts/eval_phase2_cascade.py \
     --config experiments/configs/phase2_cascade.yaml \
     --ckpt experiments/runs/phase2_cascade_vavg/checkpoints/epoch_last.pt \
     --split val
   ```

### 步骤 2: 运行方案二 (End-to-End)

无需 RoNIN 权重，直接训练。

1. **训练**:
   ```bash
   python scripts/train_phase3_end2end.py --config experiments/configs/phase3_end2end.yaml
   ```
2. **评估**:
   ```bash
   python scripts/eval_phase3_end2end.py \
     --config experiments/configs/phase3_end2end.yaml \
     --ckpt experiments/runs/phase3_end2end/checkpoints/epoch_last.pt \
     --split val
   ```

## 📂 项目结构

```
DiffusionIO/
├── data/               # 数据加载与预处理
├── models/             # 模型定义 (Diffusion, RoNIN, Hybrid)
├── training/           # 训练循环实现
├── evaluation/         # 评估脚本
├── experiments/        # 配置文件 (configs/) 与运行日志 (runs/)
└── scripts/            # CLI 入口脚本
```

## 参考文献

- **RoNIN**: [Sachini/ronin](https://github.com/Sachini/ronin)
- **DDPM**: Denoising Diffusion Probabilistic Models
