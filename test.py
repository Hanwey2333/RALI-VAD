import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import WVAD_Model
from dataset import UCFDataset

def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"Tensor {name} contains NaN")
    if torch.isinf(tensor).any():
        raise ValueError(f"Tensor {name} contains Inf")

def test(model, testloader, clip_len, gt_path, device):
    
    gt = np.load(gt_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, item in enumerate(testloader):
            clip_feature, I3D_feature, video_label, length = item if len(item) == 4 else item[1:5]
            clip_feature = clip_feature.squeeze(0)
            I3D_feature = I3D_feature.squeeze(0)

            length = int(length)
            len_cur = length

            # 这里是因为测试集的视频是按照clip_len的程度分割的，一个视频分为length/clip_len+1个batch,
            # 如果小于clip_len，则batch要设置为1，也就是unsqueeze(0)
            if len_cur < clip_len:
                clip_feature = clip_feature.unsqueeze(0)
                I3D_feature = I3D_feature.unsqueeze(0)

            lengths = torch.zeros(int(length / clip_len) + 1)
            for j in range(int(length / clip_len) + 1):
                if j == 0 and length < clip_len:
                    lengths[j] = length
                elif j == 0 and length > clip_len:
                    lengths[j] = clip_len
                    length -= clip_len
                elif length > clip_len:
                    lengths[j] = clip_len
                    length -= clip_len
                else:
                    lengths[j] = length
            lengths = lengths.to(int)

            logits1 = model(clip_feature, I3D_feature, lengths)

            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            # 将填充0的数据部分的概率剪切，恢复测试集视频原本的长度
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)

    ap1 = ap1.cpu().numpy()
    ap1 = ap1.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 15))
    AP1 = average_precision_score(gt, np.repeat(ap1, 15))

    print("AUC1: ", ROC1, " AP1: ", AP1)

    return ROC1, AP1

def ano_test(model, testloader, clip_len, gt_path, device):
    
    gt = np.load(gt_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, item in enumerate(testloader):
            clip_feature, I3D_feature, video_label, length = item if len(item) == 4 else item[1:5]
            clip_feature = clip_feature.squeeze(0)
            I3D_feature = I3D_feature.squeeze(0)

            length = int(length)
            len_cur = length

            # 这里是因为测试集的视频是按照clip_len的程度分割的，一个视频分为length/clip_len+1个batch,
            # 如果小于clip_len，则batch要设置为1，也就是unsqueeze(0)
            if len_cur < clip_len:
                clip_feature = clip_feature.unsqueeze(0)
                I3D_feature = I3D_feature.unsqueeze(0)

            lengths = torch.zeros(int(length / clip_len) + 1)
            for j in range(int(length / clip_len) + 1):
                if j == 0 and length < clip_len:
                    lengths[j] = length
                elif j == 0 and length > clip_len:
                    lengths[j] = clip_len
                    length -= clip_len
                elif length > clip_len:
                    lengths[j] = clip_len
                    length -= clip_len
                else:
                    lengths[j] = length
            lengths = lengths.to(int)

            logits1 = model(clip_feature, I3D_feature, lengths)

            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            # 将填充0的数据部分的概率剪切，恢复测试集视频原本的长度
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)

    ap1 = ap1.cpu().numpy()
    ap1 = ap1.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 15))
    AP1 = average_precision_score(gt, np.repeat(ap1, 15))

    ap1 = np.repeat(ap1, 15)

    ano_gt =  np.concatenate([gt[0:450960], gt[-11040:]])
    ano_ap1 = np.concatenate([ap1[0:450960], ap1[-11040:]])

    ROC2 = roc_auc_score(ano_gt, ano_ap1)

    print("AUC1: ", ROC1, " AP1: ", AP1, "ANO_AUC: ", ROC2)

    return ROC1, AP1, ROC2