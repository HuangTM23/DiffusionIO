"""Train Scheme 2 (End-to-End Diffusion).

Target: v_avg sequence directly from IMU.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from data.loaders.ronin_loader import get_ronin_dataset_from_list
from models.diffusion import ConditionalDiffusionModel, UNet1D
from models.diffusion.end2end import EndToEndDiffusion
from utils.config import ExperimentConfig, load_config_from_yaml
from utils.common import set_seed
from utils.logger import WandBLogger


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def build_dataloaders(
    cfg: ExperimentConfig,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    data = cfg.data
    train_list = data.train_list or str(Path(data.ronin_lists_root) / "list_train.txt")
    val_list = data.val_list
    test_list = data.test_list

    dataset_kwargs = dict(
        window_size=data.window_size,
        step_size=data.step_size,
        dataset_type=data.dataset_type,
        velocity_interval=data.velocity_interval,
    )

    train_ds = get_ronin_dataset_from_list(
        root_dir=data.data_root,
        list_path=train_list,
        mode="train",
        **dataset_kwargs,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        drop_last=True,
    )

    val_loader = None
    if val_list:
        val_ds = get_ronin_dataset_from_list(
            root_dir=data.data_root,
            list_path=val_list,
            mode="val",
            **dataset_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=False,
        )

    test_loader = None
    if test_list:
        test_ds = get_ronin_dataset_from_list(
            root_dir=data.data_root,
            list_path=test_list,
            mode="test",
            **dataset_kwargs,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader


def build_model(cfg: ExperimentConfig, device: torch.device) -> EndToEndDiffusion:
    unet = UNet1D(
        in_channels=cfg.model.velocity_channels,
        out_channels=cfg.model.velocity_channels,
        condition_channels=cfg.model.condition_channels,
        base_channels=cfg.model.unet_base_channels,
        channel_mults=tuple(cfg.model.unet_channel_mults),
        time_emb_dim=cfg.model.unet_time_emb_dim,
    )

    diffusion = ConditionalDiffusionModel(
        denoising_network=unet,
        timesteps=cfg.model.diffusion_timesteps,
        beta_schedule=cfg.model.noise_schedule,
    )

    model = EndToEndDiffusion(diffusion=diffusion)
    model.to(device)
    return model


@torch.no_grad()
def _eval_epoch(
    model: EndToEndDiffusion, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    losses = []
    for feat, targ, _, _ in loader:
        imu = feat.to(device)
        vel_gt = targ.to(device)
        if vel_gt.ndim == 2:
            vel_gt = vel_gt.unsqueeze(-1)
        out = model(imu=imu, vel_gt=vel_gt)
        losses.append(out.loss.detach().item())
    return float(sum(losses) / max(1, len(losses)))


def train(config_path: str) -> Path:
    cfg = load_config_from_yaml(config_path)
    set_seed(cfg.seed)

    device = _resolve_device(cfg.training.device)
    print(f"Using device: {device}")

    wandb_logger = WandBLogger(
        project=cfg.training.wandb_project,
        entity=cfg.training.wandb_entity,
        config=asdict(cfg),
        name=cfg.name,
        enabled=cfg.training.use_wandb,
    )

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg, device)

    optimizer = torch.optim.AdamW(
        model.diffusion.denoising_network.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    run_dir = Path(cfg.training.output_dir) / cfg.name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val = None
    for epoch in range(cfg.training.num_epochs):
        model.train()
        for batch_id, (feat, targ, _, _) in enumerate(train_loader):
            imu = feat.to(device)
            vel_gt = targ.to(device)
            if vel_gt.ndim == 2:
                vel_gt = vel_gt.unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            out = model(imu=imu, vel_gt=vel_gt)
            out.loss.backward()

            if cfg.training.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.diffusion.denoising_network.parameters(),
                    max_norm=float(cfg.training.gradient_clip),
                )

            optimizer.step()
            global_step += 1

            if global_step % cfg.training.log_interval == 0:
                print(
                    f"epoch={epoch} step={global_step} batch={batch_id} loss={out.loss.item():.6f}"
                )
                wandb_logger.log_metrics(
                    {
                        "train/loss": out.loss.item(),
                        "train/epoch": epoch,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            if (
                cfg.training.max_steps is not None
                and global_step >= cfg.training.max_steps
            ):
                break

        val_loss = None
        if val_loader is not None:
            val_loss = _eval_epoch(model, val_loader, device)
            wandb_logger.log_metrics(
                {"val/loss": val_loss, "val/epoch": epoch}, step=global_step
            )
            if best_val is None or val_loss < best_val:
                best_val = val_loss

        ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.diffusion.denoising_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cfg),
                "val_loss": val_loss,
                "best_val": best_val,
            },
            ckpt_path,
        )

        if cfg.training.max_steps is not None and global_step >= cfg.training.max_steps:
            break

    wandb_logger.finish()
    return ckpt_path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    ckpt = train(args.config)
    print(f"checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
