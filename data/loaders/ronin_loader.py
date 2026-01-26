# ruff: noqa: E402

"""data.loaders.ronin_loader

Lightweight wrappers around the original RoNIN data pipeline.

Key shapes:
- strided dataset: feat (6, T), targ (2,)
- seq2seq dataset: feat (6, T), targ (2, T)
"""

import math
import os
import sys

from torch.utils.data import Dataset

# RoNIN source files use absolute imports like `import data_utils`.
# Ensure the vendored RoNIN `source/` directory is on sys.path.
_RONIN_SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "external",
    "ronin",
    "source",
)
if _RONIN_SOURCE_DIR not in sys.path:
    # Keep repo root earlier in sys.path to avoid shadowing our `utils/` package
    # with RoNIN's `source/utils.py`.
    sys.path.append(_RONIN_SOURCE_DIR)

from external.ronin.source.data_glob_speed import (  # noqa: E402
    GlobSpeedSequence,
    SequenceToSequenceDataset,
    StridedSequenceDataset,
)
from external.ronin.source.transformations import RandomHoriRotate  # noqa: E402


class _RoninSeq2SeqChannelFirst(Dataset):
    """Adapter to make RoNIN seq2seq dataset channel-first.

    RoNIN's SequenceToSequenceDataset returns:
    - feat: (T, 6)
    - targ: (T, 2)

    We convert to:
    - feat: (6, T)
    - targ: (2, T)
    """

    def __init__(self, base_dataset):
        self._base = base_dataset
        self.feature_dim = getattr(base_dataset, "feature_dim", 6)
        self.target_dim = getattr(base_dataset, "target_dim", 2)

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        feat, targ, seq_id, frame_id = self._base[idx]
        # feat: (T, 6) -> (6, T)
        feat = feat.astype("float32").T
        # targ: (T, 2) -> (2, T)
        targ = targ.astype("float32").T
        return feat, targ, seq_id, frame_id


def get_ronin_dataset(
    root_dir,
    data_list,
    mode="train",
    window_size=200,
    step_size=10,
    dataset_type="strided",
    velocity_interval=None,
    max_ori_error=20.0,
    cache_path=None,
):
    random_shift = 0
    shuffle = False
    transforms = None
    grv_only = False

    if mode == "train":
        random_shift = step_size // 2
        shuffle = True
        # NOTE: RandomHoriRotate only supports (feat, targ) where targ is (2,).
        # For seq2seq we disable transforms by default.
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == "val":
        shuffle = True
    elif mode == "test":
        shuffle = False
        grv_only = True

    kwargs = {
        "grv_only": grv_only,
        "max_ori_error": max_ori_error,
    }
    if velocity_interval is not None:
        # RoNIN uses `interval` (w) to compute v = (pos[t+w]-pos[t])/dt.
        kwargs["interval"] = int(velocity_interval)

    if dataset_type == "strided":
        dataset = StridedSequenceDataset(
            GlobSpeedSequence,
            root_dir,
            data_list,
            cache_path,
            step_size,
            window_size,
            random_shift=random_shift,
            transform=transforms,
            shuffle=shuffle,
            **kwargs,
        )
        return dataset

    if dataset_type == "seq2seq":
        # SequenceToSequenceDataset returns window-wise sequences for both feat and targ.
        # We keep transforms disabled by default because RoNIN's RandomHoriRotate is not seq-aware.
        base = SequenceToSequenceDataset(
            GlobSpeedSequence,
            root_dir,
            data_list,
            cache_path,
            step_size,
            window_size,
            random_shift=random_shift,
            transform=None,
            shuffle=shuffle,
            **kwargs,
        )
        return _RoninSeq2SeqChannelFirst(base)

    raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return dataset


def get_ronin_dataset_from_list(
    root_dir,
    list_path,
    mode="train",
    window_size=200,
    step_size=10,
    dataset_type="strided",
    velocity_interval=None,
    max_ori_error=20.0,
    cache_path=None,
):
    with open(list_path) as f:
        data_list = [
            s.strip().split("," or " ")[0]
            for s in f.readlines()
            if len(s) > 0 and s[0] != "#"
        ]

    return get_ronin_dataset(
        root_dir=root_dir,
        data_list=data_list,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
        dataset_type=dataset_type,
        velocity_interval=velocity_interval,
        max_ori_error=max_ori_error,
        cache_path=cache_path,
    )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_list = "external/ronin/lists/list_train.txt"
    root_dir = "/home/dawn/datas/IO_Datasets/FRDR_RONIN/train"

    if not os.path.exists(root_dir):
        print(f"Dataset not found at {root_dir}")
        print("This is expected - dataset needs to be extracted first")
        print("Please extract the FRDR dataset zip files")
    else:
        try:
            dataset = get_ronin_dataset_from_list(
                root_dir, train_list, mode="train", window_size=200, step_size=10
            )

            print("Dataset loaded successfully")
            print(f"Number of samples: {len(dataset)}")
            print(f"Feature dim: {dataset.feature_dim}")
            print(f"Target dim: {dataset.target_dim}")

            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            batch = next(iter(dataloader))
            feat, targ, _, _ = batch

            print(f"Feature shape: {feat.shape}")
            print(f"Target shape: {targ.shape}")
            print("✓ Data loader test passed")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("This is expected if dataset is not extracted yet")
