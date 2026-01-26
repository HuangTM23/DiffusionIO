"""RoNIN + Diffusion cascade model.

This module wires a window-level RoNIN velocity estimate as an additional condition
for a conditional diffusion model that predicts a velocity sequence.

Shapes (channel-first):
- imu: (B, 6, T)
- vel_gt: (B, 2, T)
- ronin_vel: (B, 2)
- condition: (B, 8, T) = concat([imu, broadcast(ronin_vel)])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.diffusion.ddpm import ConditionalDiffusionModel


@dataclass(frozen=True)
class CascadeOutputs:
    loss: Optional[torch.Tensor] = None
    loss_per_sample: Optional[torch.Tensor] = None
    ronin_vel: Optional[torch.Tensor] = None
    pred_vel: Optional[torch.Tensor] = None
    condition: Optional[torch.Tensor] = None


class CascadeRoninDiffusion(nn.Module):
    def __init__(
        self,
        ronin: nn.Module,
        diffusion: ConditionalDiffusionModel,
        freeze_ronin: bool = True,
    ):
        super().__init__()
        self.ronin = ronin
        self.diffusion = diffusion

        if freeze_ronin:
            self.ronin.eval()
            for p in self.ronin.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _broadcast_ronin_vel(ronin_vel: torch.Tensor, seq_len: int) -> torch.Tensor:
        # ronin_vel: (B, 2) -> (B, 2, T)
        return ronin_vel.unsqueeze(-1).expand(-1, -1, seq_len)

    def build_condition(
        self, imu: torch.Tensor, ronin_vel: torch.Tensor
    ) -> torch.Tensor:
        if imu.ndim != 3:
            raise ValueError(
                f"Expected imu to be (B,6,T), got shape={tuple(imu.shape)}"
            )
        if ronin_vel.ndim != 2:
            raise ValueError(
                f"Expected ronin_vel to be (B,2), got shape={tuple(ronin_vel.shape)}"
            )
        if imu.shape[0] != ronin_vel.shape[0]:
            raise ValueError("Batch size mismatch between imu and ronin_vel")
        if ronin_vel.shape[1] != 2:
            raise ValueError("ronin_vel must have 2 channels (vx, vy)")

        seq_len = imu.shape[-1]
        ronin_seq = self._broadcast_ronin_vel(ronin_vel, seq_len)
        return torch.cat([imu, ronin_seq], dim=1)

    @torch.no_grad()
    def infer_ronin(self, imu: torch.Tensor) -> torch.Tensor:
        # RoNIN model expects (B,6,T) and returns (B,2)
        return self.ronin(imu)

    def forward(
        self,
        imu: torch.Tensor,
        vel_gt: Optional[torch.Tensor] = None,
        ronin_vel: Optional[torch.Tensor] = None,
    ) -> CascadeOutputs:
        """If vel_gt is provided, returns training losses; else only returns condition/ronin."""
        if ronin_vel is None:
            ronin_vel = self.infer_ronin(imu)

        condition = self.build_condition(imu, ronin_vel)

        if vel_gt is None:
            return CascadeOutputs(ronin_vel=ronin_vel, condition=condition)

        if vel_gt.ndim != 3:
            raise ValueError(
                f"Expected vel_gt to be (B,2,T), got shape={tuple(vel_gt.shape)}"
            )
        if vel_gt.shape[0] != imu.shape[0] or vel_gt.shape[-1] != imu.shape[-1]:
            raise ValueError("vel_gt must match imu batch size and seq_len")
        if vel_gt.shape[1] != 2:
            raise ValueError("vel_gt must have 2 channels (vx, vy)")

        loss_per_sample = self.diffusion.compute_loss(vel_gt, condition)
        loss = loss_per_sample.mean()

        return CascadeOutputs(
            loss=loss,
            loss_per_sample=loss_per_sample,
            ronin_vel=ronin_vel,
            condition=condition,
        )

    @torch.no_grad()
    def sample(
        self,
        imu: torch.Tensor,
        ronin_vel: Optional[torch.Tensor] = None,
        progress: bool = False,
    ) -> Tuple[torch.Tensor, CascadeOutputs]:
        if ronin_vel is None:
            ronin_vel = self.infer_ronin(imu)
        condition = self.build_condition(imu, ronin_vel)
        batch_size, _, seq_len = imu.shape
        pred = self.diffusion.sample(
            batch_size=batch_size,
            seq_len=seq_len,
            output_dim=2,
            condition=condition,
            device=imu.device,
            progress=progress,
        )
        return pred, CascadeOutputs(
            ronin_vel=ronin_vel, pred_vel=pred, condition=condition
        )
