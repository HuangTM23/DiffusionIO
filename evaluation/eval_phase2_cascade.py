"""Evaluate Scheme 1 (RoNIN prior + diffusion) on v_avg target."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import json

from training.train_phase2_cascade import (
    build_dataloaders,
    build_model,
    _resolve_device,
)
from utils.config import load_config_from_yaml
from utils.logger import WandBLogger


def _load_unet_state(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must be a dict with key 'model_state_dict'")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)


def _rmse_mae(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    # pred/gt: (N,2,T)
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return rmse, mae


@torch.no_grad()
def evaluate(config_path: str, ckpt_path: str, split: str = "val") -> Path:
    cfg = load_config_from_yaml(config_path)
    device = _resolve_device(cfg.training.device)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")

    # Initialize WandB for evaluation
    wandb_logger = WandBLogger(
        project=cfg.training.wandb_project,
        entity=cfg.training.wandb_entity,
        config=asdict(cfg),
        name=f"{cfg.name}_eval_{split}",
        enabled=cfg.training.use_wandb,
    )

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    if split == "train":
        loader = train_loader
    elif split == "val":
        if val_loader is None:
            raise ValueError("val_list is not set in config")
        loader = val_loader
    elif split == "test":
        if test_loader is None:
            raise ValueError("test_list is not set in config")
        loader = test_loader
    else:
        raise ValueError("split must be one of: train, val, test")

    model = build_model(cfg, device)
    _load_unet_state(model.diffusion.denoising_network, ckpt_path)
    model.eval()

    preds = []
    gts = []
    for feat, targ, _, _ in loader:
        imu = feat.to(device)
        vel_gt = targ.to(device)
        if vel_gt.ndim == 2:
            vel_gt = vel_gt.unsqueeze(-1)

        pred, _ = model.sample(imu, progress=False)
        preds.append(pred.detach().cpu().numpy())
        gts.append(vel_gt.detach().cpu().numpy())

    pred_all = np.concatenate(preds, axis=0)
    gt_all = np.concatenate(gts, axis=0)
    rmse, mae = _rmse_mae(pred_all, gt_all)
    print(f"split={split} rmse={rmse:.6f} mae={mae:.6f}")

    wandb_logger.log_metrics(
        {f"eval/{split}/rmse": rmse, f"eval/{split}/mae": mae}, step=0
    )

    run_dir = Path(cfg.training.output_dir) / cfg.name
    out_dir = run_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.npz"
    np.savez_compressed(
        out_path,
        pred_vel=pred_all,
        gt_vel=gt_all,
        rmse=rmse,
        mae=mae,
        config_json=json.dumps(asdict(cfg)),
        ckpt_path=str(ckpt_path),
    )
    
    wandb_logger.finish()
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    args = parser.parse_args(argv)

    # Load config to get defaults
    cfg = load_config_from_yaml(args.config)
    
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = cfg.evaluation.ckpt_path
    if ckpt_path is None:
        parser.error("ckpt path must be specified via --ckpt or in config file under 'evaluation.ckpt_path'")
        
    split = args.split
    if split is None:
        split = cfg.evaluation.split
        
    # Note: evaluate() re-loads config, but that's fine.
    out = evaluate(args.config, ckpt_path, split=split)
    print(f"predictions: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
