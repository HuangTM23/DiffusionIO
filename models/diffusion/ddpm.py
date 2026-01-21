"""
扩散模型基础框架
实现DDPM (Denoising Diffusion Probabilistic Models)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def linear_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionProcess(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.timesteps = timesteps

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ):
        pred_noise = model(x_t, t, condition)
        x_start = self.predict_start_from_noise(x_t, t, pred_noise)

        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior_mean_variance(x_start, x_t, t)
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ):
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, condition, clip_denoised
        )
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ):
        if device is None:
            device = next(model.parameters()).device

        x_t = torch.randn(shape, device=device)

        timesteps_range = range(self.timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm

            timesteps_range = tqdm(timesteps_range, desc="Sampling")

        for t_idx in timesteps_range:
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, condition, clip_denoised=True)

        return x_t

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise)
        pred_noise = model(x_t, t, condition)

        loss = F.mse_loss(pred_noise, noise, reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        return loss

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        denoising_network: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.denoising_network = denoising_network
        self.diffusion = DiffusionProcess(
            timesteps, beta_schedule, beta_start, beta_end
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ):
        return self.denoising_network(x_t, t, condition)

    def compute_loss(
        self, x_start: torch.Tensor, condition: Optional[torch.Tensor] = None
    ):
        batch_size = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=device)
        return self.diffusion.training_losses(self, x_start, t, condition)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        output_dim: int,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
    ):
        shape = (batch_size, output_dim, seq_len)
        return self.diffusion.p_sample_loop(self, shape, condition, device, progress)


if __name__ == "__main__":
    from unet1d import UNet1D

    batch_size = 4
    seq_len = 200
    input_dim = 2
    output_dim = 2
    condition_dim = 6
    timesteps = 1000

    unet = UNet1D(
        in_channels=input_dim,
        out_channels=output_dim,
        condition_channels=condition_dim,
        base_channels=64,
    )

    model = ConditionalDiffusionModel(
        denoising_network=unet, timesteps=timesteps, beta_schedule="linear"
    )

    x_start = torch.randn(batch_size, output_dim, seq_len)
    condition = torch.randn(batch_size, condition_dim, seq_len)

    loss = model.compute_loss(x_start, condition)
    print(f"Loss shape: {loss.shape}")
    print(f"Mean loss: {loss.mean().item():.6f}")

    samples = model.sample(
        batch_size=2,
        seq_len=seq_len,
        output_dim=output_dim,
        condition=condition[:2],
        progress=False,
    )
    print(f"Generated samples shape: {samples.shape}")
