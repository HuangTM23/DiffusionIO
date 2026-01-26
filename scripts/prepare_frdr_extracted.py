#!/usr/bin/env python3

"""Extract FRDR RoNIN dataset zips into a single directory.

This repo stores FRDR as multiple zip files under:
- data/datasets/FRDR/Data/*.zip

Each zip already contains RoNIN-compatible per-sequence folders:
- <seq_name>/info.json
- <seq_name>/data.hdf5

This script:
1) Extracts those sequence folders into one output directory.
2) Optionally verifies a few extracted sequences (info.json + key HDF5 paths).
3) Optionally deletes the original zip files to reduce disk usage.

Safety:
- Deletion is opt-in via --delete-zips.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import zipfile
from typing import List, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _check_one_sequence(seq_dir: str) -> Optional[str]:
    info_path = os.path.join(seq_dir, "info.json")
    h5_path = os.path.join(seq_dir, "data.hdf5")
    if not os.path.isfile(info_path):
        return "missing info.json"
    if not os.path.isfile(h5_path):
        return "missing data.hdf5"

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if not isinstance(info, dict):
            return "info.json is not a JSON object"
    except Exception as e:
        return f"info.json parse failed: {e}"

    try:
        import h5py

        with h5py.File(h5_path, "r") as f:
            required = [
                "synced/time",
                "synced/gyro_uncalib",
                "synced/acce",
                "pose/tango_pos",
                "pose/tango_ori",
            ]
            for key in required:
                if key not in f:
                    return f"missing HDF5 key: {key}"
    except Exception as e:
        return f"data.hdf5 open/check failed: {e}"

    return None


def extract_zip(zip_path: str, out_dir: str) -> None:
    # zip already contains top-level seq folders.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zips-dir",
        default="data/datasets/FRDR/Data",
        help="Directory containing FRDR zip files",
    )
    parser.add_argument(
        "--out-dir",
        default="data/datasets/FRDR/extracted",
        help="Directory to extract sequences into",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=3,
        help="Verify N extracted sequences after extraction (0 disables)",
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete the original zip files after successful extraction+verification",
    )
    parser.add_argument(
        "--clean-out-dir",
        action="store_true",
        help="Delete out-dir before extracting (DANGEROUS)",
    )
    args = parser.parse_args(argv)

    zip_paths = sorted(glob.glob(os.path.join(args.zips_dir, "*.zip")))
    if not zip_paths:
        print(f"No zip files found under: {args.zips_dir}")
        return 2

    if args.clean_out_dir and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)

    _ensure_dir(args.out_dir)

    print(f"Extracting {len(zip_paths)} zip(s) into: {args.out_dir}")
    for zp in zip_paths:
        print(f"- {os.path.basename(zp)}")
        extract_zip(zp, args.out_dir)

    # verification: sample a few sequence dirs
    if args.verify_samples > 0:
        seq_dirs = sorted(
            d
            for d in os.listdir(args.out_dir)
            if os.path.isdir(os.path.join(args.out_dir, d))
        )
        if not seq_dirs:
            print("No sequence directories found after extraction")
            return 1

        print(
            f"Verifying {min(args.verify_samples, len(seq_dirs))} extracted sequence(s)..."
        )
        for seq in seq_dirs[: args.verify_samples]:
            err = _check_one_sequence(os.path.join(args.out_dir, seq))
            if err:
                print(f"✗ {seq}: {err}")
                return 1
            print(f"✓ {seq}")

    if args.delete_zips:
        print("Deleting original zip files...")
        for zp in zip_paths:
            os.remove(zp)
        print("✓ zip files deleted")

    print("✓ extraction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
