import os
import torch
import numpy as np
import random


from torch.utils.tensorboard import SummaryWriter

from model import WVAD_Model
from test import *
from train import train_ucf
from dataset import UCFDataset
from torch.utils.data import DataLoader
import option

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    writer = SummaryWriter(log_dir='./runs/' + args.training_name)

    setup_seed(args.seed)

    normal_dataset = UCFDataset(args.visual_length, args.dataset, False, True, args.feature_mode)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    anomaly_dataset = UCFDataset(args.visual_length, args.dataset, False, False, args.feature_mode)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.dataset, True, False, args.feature_mode)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = WVAD_Model(embed_dim=args.embed_dim, visual_length=args.visual_length, visual_head=args.visual_head, visual_layers=args.visual_layers, attn_window=args.attn_window, devices=device)

    AUC, AP = test(model, test_loader, args.visual_length, args.gt_path, device)

    print("Initial AUC:", AUC, "AP:", AP)

    train_ucf(model, normal_loader, anomaly_loader, test_loader, args, writer, device)

    writer.close()
