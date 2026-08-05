import numpy as np
import torch
import torch.utils.data as data
import json
from torch.utils.data import DataLoader
import pandas as pd

from model import WVAD_Model

from utils import *


def load_feature_any(path: str, key: str = "features"):
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        return arr[key]
    return arr

def apply_feature_mode(clip_feature, I3D_feature, feature_mode: str):
    if feature_mode == "clip":
        I3D_feature = np.zeros_like(I3D_feature)
    elif feature_mode == "i3d":
        clip_feature = np.zeros_like(clip_feature)
    elif feature_mode != "clip_i3d":
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    return clip_feature, I3D_feature


class UCFDataset(data.Dataset):
    def __init__(self, clip_num: int, file_path: str, test_mode: bool, normal: bool = False, feature_mode: str = "clip_i3d"):

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.train_list = data["train"]
        self.test_list = data["test"]
        self.clip_num = clip_num
        self.test_mode = test_mode
        self.normal = normal
        self.feature_mode = feature_mode

        if test_mode == True:
            self.list = self.test_list
        else:
            if normal == True:
                self.list = self.train_list["Normal"]
            else:
                self.list = self.train_list["Abnormal"]
        
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        clip_feature = load_feature_any(self.list[index]["clip_path"])
        I3D_feature = load_feature_any(self.list[index]["I3D_path"]).reshape(-1,2048)

        expected=int(self.list[index]["clip_num"]); clip_feature=np.pad(clip_feature,((0,max(0,expected-clip_feature.shape[0])),(0,0))); I3D_feature=np.pad(I3D_feature,((0,max(0,expected-I3D_feature.shape[0])),(0,0))); clip_feature=clip_feature[:expected,:]; I3D_feature=I3D_feature[:expected,:]
        clip_feature, I3D_feature = apply_feature_mode(clip_feature, I3D_feature, self.feature_mode)

        if self.test_mode == False:
            clip_feature, clip_length = process_feat(clip_feature, self.clip_num)
            I3D_feature, I3D_length = process_feat(I3D_feature, self.clip_num)
        else:
            clip_feature, clip_length = process_split(clip_feature, self.clip_num)
            I3D_feature, I3D_length = process_split(I3D_feature, self.clip_num)

        assert I3D_feature.shape[0] == clip_feature.shape[0]

        clip_feature = torch.tensor(clip_feature)
        I3D_feature = torch.tensor(I3D_feature)
        video_label = self.list[index]["video_label"]
        return clip_feature, I3D_feature, video_label, clip_length



class UCF_SVAD_Dataset(data.Dataset):
    def __init__(self, clip_num: int, file_path: str, test_mode: bool, normal: bool = False, feature_mode: str = "clip_i3d"):

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.test_list = data["test"]

        with open(file_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        
        self.train_list = data2["train"]


        self.clip_num = clip_num
        self.test_mode = test_mode
        self.normal = normal
        self.feature_mode = feature_mode

        if test_mode == True:
            self.list = self.test_list
        else:
            if normal == True:
                self.list = self.train_list["Normal"]
            else:
                self.list = self.train_list["Abnormal"]
        
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        clip_feature = load_feature_any(self.list[index]["clip_path"])
        I3D_feature = load_feature_any(self.list[index]["I3D_path"]).reshape(-1,2048)

        expected=int(self.list[index]["clip_num"]); clip_feature=np.pad(clip_feature,((0,max(0,expected-clip_feature.shape[0])),(0,0))); I3D_feature=np.pad(I3D_feature,((0,max(0,expected-I3D_feature.shape[0])),(0,0))); clip_feature=clip_feature[:expected,:]; I3D_feature=I3D_feature[:expected,:]
        clip_feature, I3D_feature = apply_feature_mode(clip_feature, I3D_feature, self.feature_mode)

        clip_label = np.zeros(self.clip_num)

        if self.test_mode == False:
            clip_label = np.array(self.list[index]["merge_label"])
            #clip_label = np.array(self.list[index]["pseudo_label"])

        if self.test_mode == False:
            clip_feature, clip_length = process_feat(clip_feature, self.clip_num)
            I3D_feature, I3D_length = process_feat(I3D_feature, self.clip_num)

            new_clip_label = np.zeros(self.clip_num).astype(np.float32)
            if len(clip_label) > self.clip_num:
                r = np.linspace(0, len(clip_label), self.clip_num+1, dtype=np.int32)
                for i in range(self.clip_num):
                    if(r[i]!=r[i+1]):
                        new_clip_label[i] = np.mean(clip_label[r[i]:r[i+1]])
                    else:
                        new_clip_label[i] = clip_label[r[i]]
            else:
                new_clip_label[:len(clip_label)] = clip_label[:len(clip_label)]
            
            clip_label = new_clip_label
        else:
            clip_feature, clip_length = process_split(clip_feature, self.clip_num)
            I3D_feature, I3D_length = process_split(I3D_feature, self.clip_num)

        assert I3D_feature.shape[0] == clip_feature.shape[0]

        clip_feature = torch.tensor(clip_feature)
        I3D_feature = torch.tensor(I3D_feature)
        video_label = self.list[index]["video_label"]
        clip_label = torch.tensor(clip_label)
        return index, clip_feature, I3D_feature, video_label, clip_length, clip_label
