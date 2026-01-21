"""
1D UNet for time series denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, out_channels), nn.ReLU())

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        time_emb = self.time_mlp(t_emb)
        x = x + time_emb[:, :, None]

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x + residual


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        condition_channels=None,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=128,
        kernel_size=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        if condition_channels is not None:
            self.cond_proj = nn.Conv1d(condition_channels, in_channels, 1)
            input_channels = in_channels * 2
        else:
            self.cond_proj = None
            input_channels = in_channels

        channels = [base_channels * m for m in channel_mults]

        self.input_conv = nn.Conv1d(
            input_channels, base_channels, kernel_size, padding=kernel_size // 2
        )

        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        in_ch = base_channels
        for ch in channels:
            self.down_blocks.append(Block1D(in_ch, ch, time_emb_dim, kernel_size))
            self.down_pools.append(nn.Conv1d(ch, ch, kernel_size=2, stride=2))
            in_ch = ch

        self.bottleneck = Block1D(channels[-1], channels[-1], time_emb_dim, kernel_size)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(len(channels) - 1, -1, -1):
            out_ch = channels[i - 1] if i > 0 else base_channels
            self.up_samples.append(
                nn.ConvTranspose1d(channels[i], channels[i], kernel_size=2, stride=2)
            )
            self.up_blocks.append(
                Block1D(channels[i] * 2, out_ch, time_emb_dim, kernel_size)
            )

        self.output_conv = nn.Sequential(
            nn.Conv1d(
                base_channels, base_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.Conv1d(base_channels, out_channels, 1),
        )

    def forward(self, x, t, condition=None):
        t_emb = self.time_mlp(t)

        if condition is not None and self.cond_proj is not None:
            cond = self.cond_proj(condition)
            x = torch.cat([x, cond], dim=1)

        x = self.input_conv(x)

        down_features = []
        for down_block, down_pool in zip(self.down_blocks, self.down_pools):
            x = down_block(x, t_emb)
            down_features.append(x)
            x = down_pool(x)

        x = self.bottleneck(x, t_emb)

        for up_sample, up_block, skip in zip(
            self.up_samples, self.up_blocks, reversed(down_features)
        ):
            x = up_sample(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, t_emb)

        x = self.output_conv(x)
        return x


if __name__ == "__main__":
    import torch.nn.functional as F

    batch_size = 4
    seq_len = 200
    in_channels = 2
    out_channels = 2
    condition_channels = 6

    model = UNet1D(
        in_channels=in_channels,
        out_channels=out_channels,
        condition_channels=condition_channels,
        base_channels=64,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    x = torch.randn(batch_size, in_channels, seq_len)
    t = torch.randint(0, 1000, (batch_size,))
    condition = torch.randn(batch_size, condition_channels, seq_len)

    output = model(x, t, condition)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Condition shape: {condition.shape}")
    assert output.shape == x.shape, "Output shape mismatch"
    print("✓ UNet1D test passed")
