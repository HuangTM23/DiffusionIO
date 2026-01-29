#!/usr/bin/env python3

"""Wrapper to run RoNIN ResNet training/testing using the original script.

This script:
- Filters RoNIN official lists against the sequences present under data_root.
- Calls `external/ronin/source/ronin_resnet.py` as-is (subprocess) with a stable cwd.

Data root must contain per-sequence folders:
  <seq>/info.json
  <seq>/data.hdf5
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RONIN_SOURCE_DIR = REPO_ROOT / "external" / "ronin" / "source"
RONIN_SCRIPT = RONIN_SOURCE_DIR / "ronin_resnet.py"
RONIN_LISTS_DIR = REPO_ROOT / "external" / "ronin" / "lists"


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(str(path))


def _sequence_names(data_root: Path) -> set:
    if not data_root.is_dir():
        raise FileNotFoundError(str(data_root))
    return {
        d.name
        for d in data_root.iterdir()
        if d.is_dir() and (d / "info.json").is_file() and (d / "data.hdf5").is_file()
    }


def _filter_list(src: Path, avail: set, dst: Path) -> int:
    _require_file(src)
    keep: List[str] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split(",")[0]
            if not s or s.startswith("#"):
                continue
            if s in avail:
                keep.append(s)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))
    return len(keep)


def _prepare_available_lists(data_root: Path, out_dir: Path) -> Dict[str, Path]:
    avail = _sequence_names(data_root)
    if not avail:
        raise RuntimeError(
            f"No sequences found under {data_root} (expected <seq>/info.json + data.hdf5)"
        )

    mapping = {
        "train": (
            RONIN_LISTS_DIR / "list_train.txt",
            out_dir / "list_train_available.txt",
        ),
        "val": (RONIN_LISTS_DIR / "list_val.txt", out_dir / "list_val_available.txt"),
        "test_seen": (
            RONIN_LISTS_DIR / "list_test_seen.txt",
            out_dir / "list_test_seen_available.txt",
        ),
        "test_unseen": (
            RONIN_LISTS_DIR / "list_test_unseen.txt",
            out_dir / "list_test_unseen_available.txt",
        ),
    }

    out: Dict[str, Path] = {}
    for key, (src, dst) in mapping.items():
        kept = _filter_list(src, avail, dst)
        print(f"{key}: kept {kept} -> {dst.relative_to(REPO_ROOT)}")
        out[key] = dst
    return out


def _run_ronin(args_list: List[str]) -> int:
    _require_file(RONIN_SCRIPT)
    cmd = ["python3", str(RONIN_SCRIPT), *args_list]
    print("cmd:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(RONIN_SOURCE_DIR))
    return int(proc.returncode)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--data-root",
            default="data/datasets/FRDR/extracted",
            help="Extracted FRDR root containing per-sequence folders",
        )
        p.add_argument(
            "--lists-out-dir",
            default="data/datasets/FRDR",
            help="Where to write *_available.txt lists",
        )
        p.add_argument("--dataset", default="ronin", choices=["ronin", "ridi"])
        p.add_argument("--max-ori-error", type=float, default=20.0)
        p.add_argument("--step-size", type=int, default=10)
        p.add_argument("--window-size", type=int, default=200)
        p.add_argument("--arch", default="resnet18")
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--cache-path", default=None)
        p.add_argument("--wandb-project", default="DiffusionIO")
        p.add_argument("--wandb-entity", default=None)

    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--out-dir", default="experiments/runs/ronin_resnet")
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--continue-from", default=None)

    p_test = sub.add_parser("test")
    add_common(p_test)
    p_test.add_argument("--out-dir", default="experiments/runs/ronin_resnet_test")
    p_test.add_argument("--model-path", required=True)
    p_test.add_argument(
        "--split",
        default="test_seen",
        choices=["test_seen", "test_unseen"],
        help="Which official test list to use",
    )
    p_test.add_argument("--fast-test", action="store_true")
    p_test.add_argument("--show-plot", action="store_true")

    p_lists = sub.add_parser("prepare-lists")
    add_common(p_lists)

    ns = parser.parse_args(argv)

    data_root = (
        (REPO_ROOT / ns.data_root).resolve()
        if not os.path.isabs(ns.data_root)
        else Path(ns.data_root)
    )
    lists_out_dir = (
        (REPO_ROOT / ns.lists_out_dir).resolve()
        if not os.path.isabs(ns.lists_out_dir)
        else Path(ns.lists_out_dir)
    )
    lists = _prepare_available_lists(data_root=data_root, out_dir=lists_out_dir)

    if ns.mode == "prepare-lists":
        return 0

    common = [
        "--dataset",
        ns.dataset,
        "--root_dir",
        str(data_root),
        "--max_ori_error",
        str(ns.max_ori_error),
        "--step_size",
        str(ns.step_size),
        "--window_size",
        str(ns.window_size),
        "--arch",
        ns.arch,
    ]
    if ns.cpu:
        common.append("--cpu")
    if ns.cache_path:
        common += ["--cache_path", ns.cache_path]
    
    # WandB args
    common += ["--wandb_project", ns.wandb_project]
    if ns.wandb_entity:
        common += ["--wandb_entity", ns.wandb_entity]

    if ns.mode == "train":
        args_list = [
            "--mode",
            "train",
            "--train_list",
            str(lists["train"]),
            "--val_list",
            str(lists["val"]),
            "--out_dir",
            str(
                (REPO_ROOT / ns.out_dir).resolve()
                if not os.path.isabs(ns.out_dir)
                else ns.out_dir
            ),
            "--lr",
            str(ns.lr),
            "--batch_size",
            str(ns.batch_size),
            "--epochs",
            str(ns.epochs),
        ]
        if ns.continue_from:
            args_list += ["--continue_from", ns.continue_from]
        return _run_ronin(common + args_list)

    # test
    test_list = lists[ns.split]
    args_list = [
        "--mode",
        "test",
        "--test_list",
        str(test_list),
        "--out_dir",
        str(
            (REPO_ROOT / ns.out_dir).resolve()
            if not os.path.isabs(ns.out_dir)
            else ns.out_dir
        ),
        "--model_path",
        str(
            (REPO_ROOT / ns.model_path).resolve()
            if not os.path.isabs(ns.model_path)
            else ns.model_path
        ),
    ]
    if ns.fast_test:
        args_list.append("--fast_test")
    if ns.show_plot:
        args_list.append("--show_plot")
    return _run_ronin(common + args_list)


if __name__ == "__main__":
    raise SystemExit(main())
