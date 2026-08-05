from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from dataset_rali_v2 import resample_sequence
from test import ano_test


def valid_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    return arange < lengths.to(device).long().unsqueeze(1)


def weighted_bce_loss(logits, targets, weights, lengths, device):
    probs = torch.sigmoid(logits).squeeze(-1)
    targets = targets.to(device).float()
    weights = weights.to(device).float()
    mask = valid_mask(lengths, probs.shape[1], device).float()
    weights = weights * mask
    loss = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), targets, reduction="none")
    denom = weights.sum().clamp_min(1e-6)
    return (loss * weights).sum() / denom, probs, weights


def weighted_entropy(probs, weights):
    entropy = -(probs * torch.log(probs.clamp_min(1e-6)) + (1 - probs) * torch.log((1 - probs).clamp_min(1e-6)))
    return (entropy * weights).sum() / weights.sum().clamp_min(1e-6)


def collect_bank_from_dataset(dataset, device):
    target_chunks = []
    reliability_chunks = []
    for item in dataset.list:
        target = torch.tensor(resample_sequence(item["merge_label"], dataset.clip_num)).reshape(1, dataset.clip_num, 1)
        reliability = torch.tensor(resample_sequence(item["rali_reliability"], dataset.clip_num)).reshape(1, dataset.clip_num, 1)
        target_chunks.append(target)
        reliability_chunks.append(reliability)
    return torch.cat(target_chunks, dim=0).to(device).float(), torch.cat(reliability_chunks, dim=0).to(device).float()


def bank_stats(target_bank, reliability_bank):
    target = target_bank.detach().float().cpu().reshape(-1)
    rel = reliability_bank.detach().float().cpu().reshape(-1)
    return {
        "target_mean": float(target.mean()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_pos_ratio_gt_0p5": float((target > 0.5).float().mean()),
        "reliability_mean": float(rel.mean()),
        "reliability_min": float(rel.min()),
        "reliability_max": float(rel.max()),
    }


def update_bank(target_bank, indices, probs, lengths, beta, device):
    drift_sum = 0.0
    count = 0
    with torch.no_grad():
        for row, dataset_idx in enumerate(indices.tolist()):
            length = int(lengths[row])
            old = target_bank[dataset_idx, :length, 0]
            new = (beta * old + (1.0 - beta) * probs[row, :length].detach()).clamp(1e-6, 1 - 1e-6)
            drift_sum += float(torch.mean(torch.abs(new - old)).detach().cpu())
            count += 1
            target_bank[dataset_idx, :length, 0] = new
    return drift_sum / max(count, 1)


def train_rali_v2(args, model, normal_loader, anomaly_loader, test_loader, device, writer):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)

    output_path = Path(args.output_path) / args.training_name
    (output_path / "checkpoint").mkdir(parents=True, exist_ok=True)
    (output_path / "log").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(output_path / "log" / "training.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    init_auc, init_ap, init_ano_auc = ano_test(model, test_loader, args.visual_length, args.gt_path, device)
    print("model initialize AUC: %.4f | AP: %.4f | Ano_AUC: %.4f" % (init_auc, init_ap, init_ano_auc))
    logging.info("initialize | AUC: %.6f | AP: %.6f | Ano_AUC: %.6f", init_auc, init_ap, init_ano_auc)

    normal_target_bank, normal_reliability_bank = collect_bank_from_dataset(normal_loader.dataset, device)
    anomaly_target_bank, anomaly_reliability_bank = collect_bank_from_dataset(anomaly_loader.dataset, device)
    initial_normal_bank = normal_target_bank.clone().detach()
    initial_anomaly_bank = anomaly_target_bank.clone().detach()

    label_stats = {
        "initial_normal": bank_stats(normal_target_bank, normal_reliability_bank),
        "initial_anomaly": bank_stats(anomaly_target_bank, anomaly_reliability_bank),
        "warmup_epoch": args.warmup_epoch,
        "beta": args.rali_beta,
        "lambda_s": args.lambda_s,
    }
    (output_path / "label_stats_runtime.json").write_text(json.dumps(label_stats, indent=2), encoding="utf-8")

    total_steps = args.max_epoch * min(len(normal_loader), len(anomaly_loader))
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    process_bar = tqdm(total=total_steps)

    best = {"auc": -1.0, "ap": -1.0, "ano_auc": -1.0, "epoch": -1, "step": -1}
    global_step = 0
    metrics = []

    for epoch in range(args.max_epoch):
        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for step in range(min(len(normal_loader), len(anomaly_loader))):
            norm_idx, norm_clip, norm_i3d, _, norm_lengths, _, _ = next(normal_iter)
            anom_idx, anom_clip, anom_i3d, _, anom_lengths, _, _ = next(anomaly_iter)
            clip_features = torch.cat([norm_clip, anom_clip], dim=0).to(device)
            i3d_features = torch.cat([norm_i3d, anom_i3d], dim=0).to(device)
            lengths = torch.cat([norm_lengths, anom_lengths], dim=0).to(device)

            norm_targets = normal_target_bank[norm_idx].squeeze(-1)
            anom_targets = anomaly_target_bank[anom_idx].squeeze(-1)
            norm_weights = normal_reliability_bank[norm_idx].squeeze(-1)
            anom_weights = anomaly_reliability_bank[anom_idx].squeeze(-1)
            targets = torch.cat([norm_targets, anom_targets], dim=0)
            weights = torch.cat([norm_weights, anom_weights], dim=0)

            logits = model(clip_features, i3d_features, lengths)
            cls_loss, probs, effective_weights = weighted_bce_loss(logits, targets, weights, lengths, device)
            if not torch.isfinite(cls_loss):
                raise ValueError("non-finite reliability-weighted BCE")
            loss = cls_loss
            ent = torch.tensor(0.0, device=device)
            if args.use_sharpness_loss:
                ent = weighted_entropy(probs, effective_weights)
                loss = loss + args.lambda_s * ent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            drift = 0.0
            if args.use_label_update and epoch >= args.warmup_epoch:
                norm_probs = probs[:len(norm_idx)]
                anom_probs = probs[len(norm_idx):]
                drift += update_bank(normal_target_bank, norm_idx, norm_probs, norm_lengths, args.rali_beta, device)
                drift += update_bank(anomaly_target_bank, anom_idx, anom_probs, anom_lengths, args.rali_beta, device)
                drift *= 0.5

            if effective_weights.sum().item() <= 0:
                raise ValueError("sum(reliability weights) is zero")

            writer.add_scalar("loss", float(loss.item()), global_step)
            writer.add_scalar("cls_loss", float(cls_loss.item()), global_step)
            writer.add_scalar("entropy", float(ent.item()), global_step)
            writer.add_scalar("target_drift", float(drift), global_step)

            eval_interval = args.eval_interval if args.eval_interval and args.eval_interval > 0 else 100
            if global_step % eval_interval == 0:
                auc, ap, ano_auc = ano_test(model, test_loader, args.visual_length, args.gt_path, device)
                row = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "step": step,
                    "auc": auc,
                    "ap": ap,
                    "ano_auc": ano_auc,
                    "loss": float(loss.item()),
                    "cls_loss": float(cls_loss.item()),
                    "entropy": float(ent.item()),
                    "target_drift": float(drift),
                    "weight_sum": float(effective_weights.sum().item()),
                }
                metrics.append(row)
                print("AUC1: ", auc, " AP1: ", ap, "ANO_AUC: ", ano_auc)
                logging.info("metrics %s", json.dumps(row))
                if auc > best["auc"]:
                    best = {"auc": auc, "ap": ap, "ano_auc": ano_auc, "epoch": epoch, "step": step, "global_step": global_step}
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best": best,
                    }, output_path / "checkpoint" / f"best_epoch{epoch}_step{step}.pth")
                (output_path / "metrics.json").write_text(json.dumps({"best": best, "history": metrics}, indent=2), encoding="utf-8")

            process_bar.update(1)
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    process_bar.close()
    torch.save(model.state_dict(), output_path / "checkpoint" / "model_final.pth")
    final_stats = {
        "final_normal": bank_stats(normal_target_bank, normal_reliability_bank),
        "final_anomaly": bank_stats(anomaly_target_bank, anomaly_reliability_bank),
        "normal_target_mean_abs_drift": float(torch.mean(torch.abs(normal_target_bank - initial_normal_bank)).detach().cpu()),
        "anomaly_target_mean_abs_drift": float(torch.mean(torch.abs(anomaly_target_bank - initial_anomaly_bank)).detach().cpu()),
    }
    label_stats.update(final_stats)
    (output_path / "label_stats_runtime.json").write_text(json.dumps(label_stats, indent=2), encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps({"best": best, "history": metrics, "target_bank": final_stats}, indent=2), encoding="utf-8")

