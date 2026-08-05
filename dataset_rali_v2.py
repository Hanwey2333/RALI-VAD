from __future__ import annotations

import json

import numpy as np
import torch
import torch.utils.data as data

from dataset import apply_feature_mode
from utils import pad, process_feat, process_split


def load_feature_any(path: str, key: str = "features"):
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        return arr[key]
    return arr


def resample_sequence(values: list[float], target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    if len(arr) == 0:
        return out
    if len(arr) > target_len:
        bins = np.linspace(0, len(arr), target_len + 1, dtype=np.int32)
        for i in range(target_len):
            if bins[i] != bins[i + 1]:
                out[i] = float(np.mean(arr[bins[i]:bins[i + 1]]))
            else:
                out[i] = float(arr[min(bins[i], len(arr) - 1)])
    else:
        out[:len(arr)] = arr
    return out


class UCF_RALI_V2_Dataset(data.Dataset):
    def __init__(self, clip_num: int, file_path: str, test_mode: bool, normal: bool = False, feature_mode: str = "clip_i3d"):
        with open(file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        self.train_list = payload["train"]
        self.test_list = payload["test"]
        self.clip_num = clip_num
        self.test_mode = test_mode
        self.normal = normal
        self.feature_mode = feature_mode
        if test_mode:
            self.list = self.test_list
        elif normal:
            self.list = self.train_list["Normal"]
        else:
            self.list = self.train_list["Abnormal"]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        item = self.list[index]
        clip_feature = load_feature_any(item["clip_path"])
        i3d_feature = load_feature_any(item["I3D_path"]).reshape(-1, 2048)

        expected = int(item["clip_num"])
        clip_feature = np.pad(clip_feature, ((0, max(0, expected - clip_feature.shape[0])), (0, 0)))
        i3d_feature = np.pad(i3d_feature, ((0, max(0, expected - i3d_feature.shape[0])), (0, 0)))
        clip_feature = clip_feature[:expected, :]
        i3d_feature = i3d_feature[:expected, :]
        clip_feature, i3d_feature = apply_feature_mode(clip_feature, i3d_feature, self.feature_mode)

        if self.test_mode:
            clip_feature, clip_length = process_split(clip_feature, self.clip_num)
            i3d_feature, i3d_length = process_split(i3d_feature, self.clip_num)
            assert i3d_feature.shape[0] == clip_feature.shape[0]
            return torch.tensor(clip_feature), torch.tensor(i3d_feature), item["video_label"], clip_length

        target = resample_sequence(item["merge_label"], self.clip_num)
        reliability = resample_sequence(item["rali_reliability"], self.clip_num)

        clip_feature, clip_length = process_feat(clip_feature, self.clip_num)
        i3d_feature, i3d_length = process_feat(i3d_feature, self.clip_num)
        assert i3d_feature.shape[0] == clip_feature.shape[0]

        return (
            index,
            torch.tensor(clip_feature),
            torch.tensor(i3d_feature),
            item["video_label"],
            clip_length,
            torch.tensor(target),
            torch.tensor(reliability),
        )

