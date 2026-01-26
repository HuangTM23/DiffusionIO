#!/usr/bin/env python3

"""Smoke test for Scheme 1 cascade wiring.

This test does not require the real dataset.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from models.diffusion import ConditionalDiffusionModel, UNet1D
from models.hybrid import CascadeRoninDiffusion
from models.ronin import get_ronin_resnet


def main() -> int:
    torch.manual_seed(0)

    batch_size = 2
    # RoNIN ResNet head uses a window_size-dependent in_dim; 200 is the repo default.
    seq_len = 200
    timesteps = 20

    ronin = get_ronin_resnet(
        arch="resnet18",
        num_inputs=6,
        num_outputs=2,
        window_size=seq_len,
    )

    unet = UNet1D(
        in_channels=2,
        out_channels=2,
        condition_channels=8,
        base_channels=32,
        channel_mults=(1, 2, 4),
        time_emb_dim=64,
    )
    diffusion = ConditionalDiffusionModel(
        unet, timesteps=timesteps, beta_schedule="linear"
    )
    model = CascadeRoninDiffusion(ronin=ronin, diffusion=diffusion, freeze_ronin=True)

    imu = torch.randn(batch_size, 6, seq_len)
    vel_gt = torch.randn(batch_size, 2, seq_len)

    out = model(imu=imu, vel_gt=vel_gt)
    assert out.condition is not None
    assert out.loss is not None
    assert out.loss_per_sample is not None
    assert out.condition.shape == (batch_size, 8, seq_len)
    assert out.loss_per_sample.shape == (batch_size,)
    assert torch.isfinite(out.loss)

    out.loss.backward()
    for p in model.diffusion.denoising_network.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    pred, aux = model.sample(imu)
    assert pred.shape == (batch_size, 2, seq_len)
    assert torch.isfinite(pred).all()
    assert aux.condition is not None and aux.condition.shape == (batch_size, 8, seq_len)

    print("✓ cascade smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
