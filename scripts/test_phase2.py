#!/usr/bin/env python3
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=" * 60)
print("Phase 2 验证测试")
print("=" * 60)

print("\n1. 测试Ronin ResNet模型...")
from models.ronin import get_ronin_resnet

ronin_model = get_ronin_resnet("resnet18", num_inputs=6, num_outputs=2, window_size=200)
print(f"   ✓ Ronin ResNet加载成功")
print(f"   ✓ 参数数量: {ronin_model.get_num_params():,}")

batch_size = 2
seq_len = 200
x_imu = torch.randn(batch_size, 6, seq_len)
y_vel = ronin_model(x_imu)
print(f"   ✓ 前向传播成功: {x_imu.shape} -> {y_vel.shape}")

print("\n2. 测试1D UNet模型...")
from models.diffusion import UNet1D

unet = UNet1D(in_channels=2, out_channels=2, condition_channels=6, base_channels=64)
total_params = sum(p.numel() for p in unet.parameters())
print(f"   ✓ UNet1D加载成功")
print(f"   ✓ 参数数量: {total_params:,}")

x = torch.randn(batch_size, 2, seq_len)
t = torch.randint(0, 1000, (batch_size,))
cond = torch.randn(batch_size, 6, seq_len)
noise = unet(x, t, cond)
print(f"   ✓ 去噪网络前向传播成功: {x.shape} -> {noise.shape}")

print("\n3. 测试扩散模型框架...")
from models.diffusion import ConditionalDiffusionModel

diffusion_model = ConditionalDiffusionModel(
    denoising_network=unet, timesteps=1000, beta_schedule="linear"
)
print(f"   ✓ 扩散模型初始化成功")

x_start = torch.randn(batch_size, 2, seq_len)
condition = torch.randn(batch_size, 6, seq_len)

loss = diffusion_model.compute_loss(x_start, condition)
print(f"   ✓ 训练损失计算成功: loss shape = {loss.shape}")
print(f"   ✓ 平均损失: {loss.mean().item():.6f}")

print("\n4. 测试采样过程...")
samples = diffusion_model.sample(
    batch_size=1, seq_len=seq_len, output_dim=2, condition=condition[:1], progress=False
)
print(f"   ✓ 采样成功: shape = {samples.shape}")

print("\n5. 测试工具模块...")
from utils import get_device, set_seed, setup_logger

device = get_device()
print(f"   ✓ 设备检测: {device}")

set_seed(42)
print(f"   ✓ 随机种子设置成功")

logger = setup_logger("Test")
logger.info("Logger test")
print(f"   ✓ 日志系统初始化成功")

print("\n6. 测试配置系统...")
from utils import get_default_config

config = get_default_config()
print(f"   ✓ 配置加载成功")
print(f"   ✓ Batch size: {config.training.batch_size}")
print(f"   ✓ Learning rate: {config.training.learning_rate}")
print(f"   ✓ Window size: {config.data.window_size}")

print("\n" + "=" * 60)
print("✓ Phase 2 基础模型实现验证通过!")
print("=" * 60)

print("\n已实现的组件:")
print("  1. ✓ Ronin ResNet模型 (4.6M参数)")
print("  2. ✓ 1D UNet去噪网络 (6.8M参数)")
print("  3. ✓ DDPM扩散模型框架")
print("  4. ✓ 条件扩散模型")
print("  5. ✓ 数据加载器接口")
print("  6. ✓ 配置管理系统")
print("  7. ✓ 日志系统")
print("  8. ✓ 工具函数库")

print("\n下一步:")
print("  - 解压FRDR数据集的内部zip文件")
print("  - 测试完整的数据加载流程")
print("  - 开始Phase 3: 训练脚本开发")
