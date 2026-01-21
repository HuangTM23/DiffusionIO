"""
Data loaders for IMU velocity estimation
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../external/ronin/source"))

from data_glob_speed import GlobSpeedSequence, StridedSequenceDataset
from transformations import RandomHoriRotate
import math


def get_ronin_dataset(
    root_dir,
    data_list,
    mode="train",
    window_size=200,
    step_size=10,
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
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == "val":
        shuffle = True
    elif mode == "test":
        shuffle = False
        grv_only = True

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
        grv_only=grv_only,
        max_ori_error=max_ori_error,
    )

    return dataset


def get_ronin_dataset_from_list(
    root_dir,
    list_path,
    mode="train",
    window_size=200,
    step_size=10,
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
        root_dir, data_list, mode, window_size, step_size, max_ori_error, cache_path
    )


if __name__ == "__main__":
    import torch
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

            print(f"Dataset loaded successfully")
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
