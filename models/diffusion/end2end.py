"""End-to-End Diffusion Model for IMU-based Velocity Estimation.

This model directly conditions the diffusion process on the raw IMU sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.diffusion.ddpm import ConditionalDiffusionModel


@dataclass(frozen=True)
class EndToEndOutputs:
    loss: Optional[torch.Tensor] = None
    loss_per_sample: Optional[torch.Tensor] = None
    pred_vel: Optional[torch.Tensor] = None
    condition: Optional[torch.Tensor] = None


class EndToEndDiffusion(nn.Module):
    def __init__(
        self,
        diffusion: ConditionalDiffusionModel,
    ):
        super().__init__()
        self.diffusion = diffusion

    def forward(
        self,
        imu: torch.Tensor,
        vel_gt: Optional[torch.Tensor] = None,
    ) -> EndToEndOutputs:
        """
        Args:
            imu: (B, 6, T)
            vel_gt: (B, 2, T)
        """
        # Condition is directly the IMU data
        condition = imu

        if vel_gt is None:
            return EndToEndOutputs(condition=condition)

        loss_per_sample = self.diffusion.compute_loss(vel_gt, condition)
        loss = loss_per_sample.mean()

        return EndToEndOutputs(
            loss=loss,
            loss_per_sample=loss_per_sample,
            condition=condition,
        )

    @torch.no_grad()
    def sample(
        self,
        imu: torch.Tensor,
        progress: bool = False,
    ) -> Tuple[torch.Tensor, EndToEndOutputs]:
        condition = imu
        batch_size, _, seq_len = imu.shape
        pred = self.diffusion.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            output_dim=2,
            condition=condition,
            device=imu.device,
            progress=progress,
        )
        return pred, EndToEndOutputs(pred_vel=pred, condition=condition)
