from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import option
from dataset_rali_v2 import UCF_RALI_V2_Dataset
from model import WVAD_Model
from train_rali_v2 import train_rali_v2


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()
    if not args.stage2_dataset:
        raise ValueError("--stage2_dataset is required")
    if not args.init_checkpoint:
        raise ValueError("--init_checkpoint is required")
    args.dataset = args.stage2_dataset
    args.lr = 1e-4

    setup_seed(args.seed)
    writer = SummaryWriter(log_dir="./runs/" + args.training_name)

    normal_dataset = UCF_RALI_V2_Dataset(args.visual_length, args.dataset, False, True, args.feature_mode)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCF_RALI_V2_Dataset(args.visual_length, args.dataset, False, False, args.feature_mode)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataset = UCF_RALI_V2_Dataset(args.visual_length, args.dataset, True, False, args.feature_mode)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = WVAD_Model(
        embed_dim=args.embed_dim,
        visual_length=args.visual_length,
        visual_head=args.visual_head,
        visual_layers=args.visual_layers,
        attn_window=args.attn_window,
        devices=device,
    )
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    train_rali_v2(args, model, normal_loader, anomaly_loader, test_loader, device, writer)
    writer.close()

