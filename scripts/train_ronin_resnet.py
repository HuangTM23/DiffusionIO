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
import yaml
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RONIN_SOURCE_DIR = REPO_ROOT / "external" / "ronin" / "source"
RONIN_SCRIPT = RONIN_SOURCE_DIR / "ronin_resnet.py"
RONIN_LISTS_DIR = REPO_ROOT / "external" / "ronin" / "lists"
WANDB_CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "wandb.yaml"


def _load_wandb_defaults() -> Dict[str, str]:
    defaults = {"project": "DiffusionIO", "entity": None, "mode": "online"}
    if WANDB_CONFIG_PATH.is_file():
        try:
            with WANDB_CONFIG_PATH.open("r") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    defaults.update(cfg)
            # Set mode environment variable immediately
            if "mode" in defaults:
                os.environ["WANDB_MODE"] = defaults["mode"]
            print(f"Loaded global WandB config for RoNIN: {defaults}")
        except Exception as e:
            print(f"Warning: Failed to load global WandB config: {e}")
    return defaults


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


def _load_config_defaults(argv: Optional[List[str]]) -> Dict:
    """Pre-scan argv for --config and load it."""
    defaults = {}
    
    # Check for --config in arguments
    config_path = None
    args = argv if argv is not None else sys.argv[1:]
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 < len(args):
            config_path = Path(args[idx + 1])
    
    if config_path and config_path.is_file():
        try:
            with config_path.open("r") as f:
                yaml_cfg = yaml.safe_load(f)
                if yaml_cfg:
                    # Flatten the nested config for argparse defaults
                    if "data" in yaml_cfg:
                        defaults.update({
                            "data_root": yaml_cfg["data"].get("data_root"),
                            "lists_out_dir": yaml_cfg["data"].get("lists_out_dir"),
                            "dataset": yaml_cfg["data"].get("dataset"),
                            "max_ori_error": yaml_cfg["data"].get("max_ori_error"),
                            "step_size": yaml_cfg["data"].get("step_size"),
                            "window_size": yaml_cfg["data"].get("window_size"),
                            "cache_path": yaml_cfg["data"].get("cache_path"),
                        })
                    if "model" in yaml_cfg:
                        defaults.update({"arch": yaml_cfg["model"].get("arch")})
                    if "training" in yaml_cfg:
                        defaults.update({
                            "out_dir": yaml_cfg["training"].get("out_dir"), # Note: this overlaps with testing out_dir
                            "lr": yaml_cfg["training"].get("lr"),
                            "batch_size": yaml_cfg["training"].get("batch_size"),
                            "epochs": yaml_cfg["training"].get("epochs"),
                            "continue_from": yaml_cfg["training"].get("continue_from"),
                            "cpu": yaml_cfg["training"].get("cpu"),
                        })
                    if "testing" in yaml_cfg:
                        # If running test mode, these might overwrite training defaults, which is intended
                        defaults.update({
                            "model_path": yaml_cfg["testing"].get("model_path"),
                            "split": yaml_cfg["testing"].get("split"),
                            "fast_test": yaml_cfg["testing"].get("fast_test"),
                            "show_plot": yaml_cfg["testing"].get("show_plot"),
                        })
                        # Handle out_dir conflict: if command is 'test', use testing.out_dir
                        if "test" in args:
                             defaults["out_dir"] = yaml_cfg["testing"].get("out_dir")
            print(f"Loaded config from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load config {config_path}: {e}")
            
    return defaults


def main(argv: Optional[List[str]] = None) -> int:
    import sys
    wandb_defaults = _load_wandb_defaults()
    config_defaults = _load_config_defaults(argv)
    
    # Merge defaults: wandb -> config -> hardcoded (handled by .get with default)
    # Actually, argparse default priority is: command line > default.
    # So we set 'default=...' to our loaded values.
    
    def get_def(key, hardcoded):
        return config_defaults.get(key, hardcoded)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="Path to YAML config file") # Add config arg so it's valid
        p.add_argument(
            "--data-root",
            default=get_def("data_root", "data/datasets/FRDR/extracted"),
            help="Extracted FRDR root containing per-sequence folders",
        )
        p.add_argument(
            "--lists-out-dir",
            default=get_def("lists_out_dir", "data/datasets/FRDR"),
            help="Where to write *_available.txt lists",
        )
        p.add_argument("--dataset", default=get_def("dataset", "ronin"), choices=["ronin", "ridi"])
        p.add_argument("--max-ori-error", type=float, default=get_def("max_ori_error", 20.0))
        p.add_argument("--step-size", type=int, default=get_def("step_size", 10))
        p.add_argument("--window-size", type=int, default=get_def("window_size", 200))
        p.add_argument("--arch", default=get_def("arch", "resnet18"))
        p.add_argument("--cpu", action="store_true") # Action store_true is hard to set default from file without custom action. 
        # Workaround: If config says cpu=True, we can't easily force it if arg is missing unless we change logic.
        # Simple fix: Assume CLI flag toggles it. For now, ignore config for boolean flags unless we change to store_true/false.
        
        p.add_argument("--cache-path", default=get_def("cache_path", None))
        p.add_argument("--wandb-project", default=wandb_defaults.get("project"))
        p.add_argument("--wandb-entity", default=wandb_defaults.get("entity"))

    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--out-dir", default=get_def("out_dir", "experiments/runs/ronin_resnet"))
    p_train.add_argument("--lr", type=float, default=get_def("lr", 1e-4))
    p_train.add_argument("--batch-size", type=int, default=get_def("batch_size", 128))
    p_train.add_argument("--epochs", type=int, default=get_def("epochs", 50))
    p_train.add_argument("--continue-from", default=get_def("continue_from", None))

    p_test = sub.add_parser("test")
    add_common(p_test)
    p_test.add_argument("--out-dir", default=get_def("out_dir", "experiments/runs/ronin_resnet_test"))
    
    # model-path is required in CLI usually, but if provided in config, we can make it optional in CLI
    model_path_default = get_def("model_path", None)
    p_test.add_argument("--model-path", required=(model_path_default is None), default=model_path_default)
    
    p_test.add_argument(
        "--split",
        default=get_def("split", "test_seen"),
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
