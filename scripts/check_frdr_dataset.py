#!/usr/bin/env python3

"""Check FRDR (RoNIN) dataset zips for expected format.

Expected ZIP structure:
- Data/*.zip
- Each zip contains multiple sequences under <seq_name>/
- Each sequence contains:
  - info.json
  - data.hdf5

We also sample-extract a few data.hdf5 files to a temporary file to validate
key HDF5 paths exist.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class ZipCheckResult:
    zip_path: str
    num_entries: int
    num_sequences: int
    missing_info: int
    missing_hdf5: int
    sampled_hdf5_ok: int
    sampled_hdf5_fail: int
    notes: List[str]


def _iter_sequences(zip_names: Sequence[str]) -> Dict[str, set]:
    """Return map: seq_name -> set of file basenames present."""
    seq_files: Dict[str, set] = {}
    for name in zip_names:
        if not name or name.endswith("/"):
            continue
        parts = name.split("/")
        if len(parts) < 2:
            continue
        seq = parts[0]
        base = parts[-1]
        if not seq:
            continue
        seq_files.setdefault(seq, set()).add(base)
    return seq_files


def _check_info_json(zf: zipfile.ZipFile, path_in_zip: str) -> Optional[str]:
    try:
        with zf.open(path_in_zip) as f:
            data = json.loads(f.read().decode("utf-8"))
        # minimal sanity
        if not isinstance(data, dict):
            return "info.json is not a JSON object"
        for key in ("imu_init_gyro_bias", "imu_acce_bias", "imu_acce_scale"):
            if key not in data:
                return f"info.json missing key: {key}"
        return None
    except Exception as e:
        return f"info.json parse failed: {e}"


def _check_hdf5_file(tmp_path: str) -> Optional[str]:
    try:
        import h5py

        with h5py.File(tmp_path, "r") as f:
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
            # shape sanity
            t_ds = f.get("synced/time")
            if not isinstance(t_ds, h5py.Dataset):
                return "synced/time is not a dataset"
            t = t_ds[()]
            if getattr(t, "ndim", None) != 1:
                return f"synced/time expected 1D, got shape={getattr(t, 'shape', None)}"
            if t.shape[0] < 10:
                return "synced/time too short"
        return None
    except Exception as e:
        return f"data.hdf5 open/check failed: {e}"


def check_zip(zip_path: str, sample_sequences: int = 2) -> ZipCheckResult:
    notes: List[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        seq_files = _iter_sequences(names)
        num_sequences = len(seq_files)

        missing_info = 0
        missing_hdf5 = 0
        for seq, files in seq_files.items():
            if "info.json" not in files:
                missing_info += 1
            if "data.hdf5" not in files:
                missing_hdf5 += 1

        sampled_hdf5_ok = 0
        sampled_hdf5_fail = 0

        # sample a few sequences for deeper checks
        for i, seq in enumerate(sorted(seq_files.keys())[: max(0, sample_sequences)]):
            info_path = f"{seq}/info.json"
            h5_path = f"{seq}/data.hdf5"

            if info_path in names:
                err = _check_info_json(zf, info_path)
                if err:
                    notes.append(f"{seq}: {err}")

            if h5_path in names:
                with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as tmp:
                    tmp.write(zf.read(h5_path))
                    tmp.flush()
                    err = _check_hdf5_file(tmp.name)
                if err:
                    sampled_hdf5_fail += 1
                    notes.append(f"{seq}: {err}")
                else:
                    sampled_hdf5_ok += 1
            else:
                sampled_hdf5_fail += 1
                notes.append(f"{seq}: data.hdf5 missing")

        return ZipCheckResult(
            zip_path=zip_path,
            num_entries=len(names),
            num_sequences=num_sequences,
            missing_info=missing_info,
            missing_hdf5=missing_hdf5,
            sampled_hdf5_ok=sampled_hdf5_ok,
            sampled_hdf5_fail=sampled_hdf5_fail,
            notes=notes,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="data/datasets/FRDR",
        help="FRDR dataset directory (contains Data/*.zip)",
    )
    parser.add_argument(
        "--sample-sequences",
        type=int,
        default=2,
        help="How many sequences to sample per zip for HDF5 validation",
    )
    args = parser.parse_args(argv)

    data_dir = os.path.join(args.dataset_dir, "Data")
    zip_paths = sorted(glob.glob(os.path.join(data_dir, "*.zip")))
    if not zip_paths:
        # Fallback: check extracted directory if zips are already deleted.
        extracted_dir = os.path.join(args.dataset_dir, "extracted")
        if not os.path.isdir(extracted_dir):
            print(f"No zip files found under: {data_dir}")
            print(f"And extracted directory not found: {extracted_dir}")
            return 2

        seq_dirs = sorted(
            d
            for d in os.listdir(extracted_dir)
            if os.path.isdir(os.path.join(extracted_dir, d))
        )
        if not seq_dirs:
            print(f"No sequence directories found under: {extracted_dir}")
            return 1

        print(f"No zips found. Checking extracted sequences under: {extracted_dir}")
        to_check = seq_dirs[: max(1, args.sample_sequences)]
        ok = True
        for seq in to_check:
            err = _check_hdf5_file(os.path.join(extracted_dir, seq, "data.hdf5"))
            if err:
                ok = False
                print(f"✗ {seq}: {err}")
            else:
                print(f"✓ {seq}")
        if ok:
            print("✓ extracted dataset looks RoNIN-compatible")
            return 0
        print("✗ extracted dataset failed checks")
        return 1

    print(f"Found {len(zip_paths)} zip(s) under: {data_dir}")
    results: List[ZipCheckResult] = []
    for zp in zip_paths:
        r = check_zip(zp, sample_sequences=args.sample_sequences)
        results.append(r)

    ok = True
    for r in results:
        print("-" * 80)
        print(os.path.basename(r.zip_path))
        print(f"  entries: {r.num_entries}")
        print(f"  sequences: {r.num_sequences}")
        print(f"  missing info.json: {r.missing_info}")
        print(f"  missing data.hdf5: {r.missing_hdf5}")
        print(
            f"  sampled hdf5 ok/fail: {r.sampled_hdf5_ok}/{r.sampled_hdf5_fail} (sample={args.sample_sequences})"
        )
        if r.missing_info or r.missing_hdf5 or r.sampled_hdf5_fail:
            ok = False
        if r.notes:
            print("  notes:")
            for n in r.notes[:10]:
                print(f"    - {n}")
            if len(r.notes) > 10:
                print(f"    - ... ({len(r.notes) - 10} more)")

    print("-" * 80)
    if ok:
        print(
            "✓ FRDR zips look RoNIN-compatible (info.json + data.hdf5 present; sampled HDF5 keys OK)"
        )
        print(
            "Next: extract zips into a directory and point YAML data_root to that extracted directory."
        )
        return 0
    print(
        "✗ FRDR zips failed some checks; see notes. You may have a corrupted or unexpected package."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
