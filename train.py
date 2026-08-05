import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

import logging
from tqdm import tqdm

from test import test

def sparsity_loss(arr, lamda2=1e-3):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth_loss(arr, lamda1=8e-4):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss

def MIL_BCE(logits, labels, lengths, device):
    # logits [B, T, 1]
    instance_logits = torch.zeros(0).to(device)
    labels = labels.to(device).float()
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    sparsity = torch.zeros(0).to(device)

    for i in range(logits.shape[0]):
        tmp, top_k_indices = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

        all_indices = torch.arange(lengths[i]).to(device)
        remaining_indices = all_indices[(all_indices.unsqueeze(1) != top_k_indices).all(1)]
        remaining = logits[i][remaining_indices]
        sparsity = torch.cat([sparsity, remaining], dim=0)

    loss_sparsity = sparsity_loss(sparsity)
    
    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss, loss_sparsity

def train_ucf(model, normal_loader, anomaly_loader, test_loader, args, writer, device):

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)

    auc_best = 0

    output_path = os.path.join(args.output_path, args.training_name)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, 'checkpoint'))
        os.makedirs(os.path.join(output_path, 'log'))
        print(f" {output_path} established")
    else:
        print(f"{output_path} already exists")

    logging.basicConfig(filename= os.path.join(output_path, 'log/traning.log'), level=logging.INFO, format = '%(asctime)s %(message)s')

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    total_steps = args.max_epoch * min(len(normal_loader), len(anomaly_loader))
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    process_bar = tqdm(total = total_steps)
    global_step = 0

    for e in range(args.max_epoch):

        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        step = 0
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            norm_clip_feature, norm_I3D_feature, norm_video_label, norm_lengths = next(normal_iter)
            anomaly_clip_feature, anomaly_I3D_feature, anomaly_video_label, anomaly_lengths = next(anomaly_iter)

            clip_features = torch.cat([norm_clip_feature, anomaly_clip_feature], dim=0).to(device)
            I3D_features = torch.cat([norm_I3D_feature, anomaly_I3D_feature], dim=0).to(device)
            video_labels = torch.cat([norm_video_label, anomaly_video_label], dim=0).to(device)
            lengths = torch.cat([norm_lengths, anomaly_lengths], dim=0).to(device)

            logits  = model(clip_features, I3D_features, lengths)

            loss, loss_sparsity = MIL_BCE(logits, video_labels, lengths, device)

            loss += loss_sparsity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

            eval_interval = args.eval_interval if args.eval_interval and args.eval_interval > 0 else 5
            if step % eval_interval == 0 and step != 0:

                AUC, AP = test(model, test_loader, args.visual_length, args.gt_path, device)

                logging.info("epoch: %d | step: %d | AUC: %.4f | loss: %.6f | sparsity: %.6f", e, step, AUC, loss.item(), loss_sparsity.item())

                writer.add_scalar('AUC', AUC, step)

                if AUC > auc_best:
                    auc_best = AUC
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'auc': auc_best}
                    
                    torch.save(checkpoint, os.path.join(output_path, 'checkpoint/best_epoch{}_step{}.pth'.format(e,step)))

            process_bar.update(1)
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break
        
        #scheduler.step()

    torch.save(model.state_dict(), os.path.join(output_path, 'checkpoint/model_final.pth'))
    process_bar.close()


    
