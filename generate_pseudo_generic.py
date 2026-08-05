
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import apply_feature_mode, load_feature_any
from model import WVAD_Model
from utils import process_split


def run_model(model, item, feature_mode: str, clip_len: int, device: str) -> list[float]:
    clip_feature = load_feature_any(item["clip_path"])
    i3d_feature = load_feature_any(item["I3D_path"]).reshape(-1, 2048)
    expected = int(item["clip_num"])
    clip_feature = np.pad(clip_feature, ((0, max(0, expected - clip_feature.shape[0])), (0, 0)))[:expected]
    i3d_feature = np.pad(i3d_feature, ((0, max(0, expected - i3d_feature.shape[0])), (0, 0)))[:expected]
    clip_feature, i3d_feature = apply_feature_mode(clip_feature, i3d_feature, feature_mode)
    clip_feature, clip_length = process_split(clip_feature, clip_len)
    i3d_feature, _ = process_split(i3d_feature, clip_len)
    length = int(clip_length)
    len_cur = length
    if len_cur < clip_len:
        clip_feature = clip_feature.reshape(1, clip_len, clip_feature.shape[-1])
        i3d_feature = i3d_feature.reshape(1, clip_len, i3d_feature.shape[-1])
    lengths = torch.zeros(int(length / clip_len) + 1, dtype=torch.int64)
    rem = length
    for j in range(len(lengths)):
        if j == 0 and rem < clip_len:
            lengths[j] = rem
        elif rem > clip_len:
            lengths[j] = clip_len
            rem -= clip_len
        else:
            lengths[j] = rem
    with torch.no_grad():
        logits = model(torch.tensor(clip_feature).to(device), torch.tensor(i3d_feature).to(device), lengths.to(device))
        logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2])
        prob = torch.sigmoid(logits[:len_cur].squeeze(-1)).cpu().numpy()
    return [round(float(x), 4) for x in prob]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--feature-mode", default="i3d", choices=["clip", "i3d", "clip_i3d"])
    ap.add_argument("--visual-length", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--visual-head", type=int, default=1)
    ap.add_argument("--visual-layers", type=int, default=2)
    ap.add_argument("--attn-window", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WVAD_Model(args.embed_dim, args.visual_length, args.visual_head, args.visual_layers, args.attn_window, device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    for split in ["Abnormal", "Normal"]:
        for item in data["train"][split]:
            item["pseudo_label"] = run_model(model, item, args.feature_mode, args.visual_length, device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
