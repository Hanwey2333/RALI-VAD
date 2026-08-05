from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rali_v2 import quantiles, rali_v2_clip


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_train_rows(payload: dict):
    for split in ("Abnormal", "Normal"):
        for row in payload["train"][split]:
            yield split, row


def annotation_value(row: dict) -> float:
    label = int(row.get("label", 0))
    confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
    return confidence if label == 1 else 0.0


def index_noise(noise_payload: dict) -> dict[str, dict]:
    return {
        row["video_name"]: dict(row)
        for row in noise_payload["train"]["Abnormal"] + noise_payload["train"]["Normal"]
    }


def overlay_annotations(noise_by_video: dict[str, dict], base_by_video: dict[str, dict], paths: list[Path]) -> dict:
    stats = Counter()
    stats["annotation_files"] = len(paths)
    for path in paths:
        for row in load_jsonl(path):
            stats["annotation_rows"] += 1
            if row.get("status") != "ok":
                stats["annotation_failed_rows"] += 1
                continue
            stats["annotation_ok_rows"] += 1
            video_name = row["video_name"]
            base = base_by_video.get(video_name)
            if base is None:
                stats["annotation_missing_video"] += 1
                continue
            noise = noise_by_video.get(video_name)
            if noise is None:
                clip_num = int(base.get("clip_num", len(base.get("pseudo_label", []))))
                noise = {
                    "video_name": video_name,
                    "video_label": int(base.get("video_label", 1)),
                    "clip_num": clip_num,
                    # None means the clip was not queried. rali_v2_clip then
                    # falls back to the Stage-1 pseudo score for abnormal videos.
                    "clip_label": [None for _ in range(clip_num)],
                    "annotation_override_meta": [],
                }
                noise_by_video[video_name] = noise
                stats["annotation_created_video"] += 1
            clip_idx = int(row["clip_idx"])
            if clip_idx < 0 or clip_idx >= len(noise["clip_label"]):
                stats["annotation_out_of_range"] += 1
                continue
            noise["clip_label"][clip_idx] = round(annotation_value(row), 4)
            noise.setdefault("annotation_override_meta", []).append({
                "clip_idx": clip_idx,
                "label": int(row.get("label", 0)),
                "confidence": float(row.get("confidence", 0.0)),
                "value": noise["clip_label"][clip_idx],
                "reason": row.get("reason", ""),
            })
            stats["annotation_applied"] += 1
    return dict(stats)


def align_noise(noise: list[float | None] | None, target_len: int) -> list[float | None] | None:
    if noise is None:
        return None
    values = [None if v is None else float(v) for v in noise]
    if len(values) >= target_len:
        return values[:target_len]
    return values + [None for _ in range(target_len - len(values))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pseudo", required=True)
    parser.add_argument("--noise-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-stats", required=True)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--r-min", type=float, default=0.05)
    parser.add_argument("--missing-abnormal-positive-discount", type=float, default=0.0)
    parser.add_argument("--annotation-jsonl", nargs="*", default=[])
    args = parser.parse_args()
    if not 0.0 <= args.missing_abnormal_positive_discount <= 1.0:
        parser.error("--missing-abnormal-positive-discount must be in [0, 1]")

    data = load_json(Path(args.base_pseudo))
    noise_by_video = index_noise(load_json(Path(args.noise_label)))
    base_by_video = {row["video_name"]: row for _, row in iter_train_rows(data)}
    annotation_stats = overlay_annotations(noise_by_video, base_by_video, [Path(p) for p in args.annotation_jsonl])

    branch_counts = Counter()
    video_counts = Counter()
    r_values = []
    y_values = []
    agreement_values = []
    missing_abnormal_reliability = []
    conflict_count = 0
    covered_count = 0
    total_count = 0

    for split, item in iter_train_rows(data):
        pseudo = [float(v) for v in item["pseudo_label"]]
        noise_row = noise_by_video.get(item["video_name"])
        noise = align_noise(noise_row["clip_label"], len(pseudo)) if noise_row is not None else None
        if noise_row is not None:
            item["noise_label"] = noise
            if "annotation_override_meta" in noise_row:
                item["annotation_override_meta"] = noise_row["annotation_override_meta"]

        y_r, rel = [], []
        p_arr, n_arr, cp_arr, cn_arr, branch_arr = [], [], [], [], []
        for idx, p in enumerate(pseudo):
            n = noise[idx] if noise is not None else None
            result = rali_v2_clip(p, n, video_label=int(item["video_label"]), alpha=args.alpha, tau=args.tau, r_min=args.r_min)
            effective_r = result.r
            if result.branch == "missing_abnormal_pseudo":
                effective_r = max(
                    args.r_min,
                    result.r * (1.0 - args.missing_abnormal_positive_discount * result.p),
                )
                missing_abnormal_reliability.append(effective_r)
            y_r.append(round(result.y_r, 6))
            rel.append(round(effective_r, 6))
            p_arr.append(round(result.p, 6))
            n_arr.append(None if result.n is None else round(result.n, 6))
            cp_arr.append(round(result.c_p, 6))
            cn_arr.append(None if result.c_n is None else round(result.c_n, 6))
            branch_arr.append(result.branch)
            branch_counts[result.branch] += 1
            video_counts[f"{split}:{'covered' if n is not None else 'missing'}"] += 1
            total_count += 1
            if n is not None:
                covered_count += 1
                agreement_values.append(1.0 - abs(result.p - float(n)))
                if abs(result.p - float(n)) > args.tau:
                    conflict_count += 1
            r_values.append(effective_r)
            y_values.append(result.y_r)

        item["merge_label"] = y_r
        item["rali_reliability"] = rel
        item["rali_v2"] = {
            "p": p_arr,
            "n": n_arr,
            "c_p": cp_arr,
            "c_n": cn_arr,
            "branch": branch_arr,
        }

    stats = {
        "alpha": args.alpha,
        "tau": args.tau,
        "r_min": args.r_min,
        "missing_abnormal_positive_discount": args.missing_abnormal_positive_discount,
        "total_clips": total_count,
        "covered_clips": covered_count,
        "coverage": covered_count / max(total_count, 1),
        "conflict_clips": conflict_count,
        "conflict_rate_on_covered": conflict_count / max(covered_count, 1),
        "branch_counts": dict(branch_counts),
        "video_clip_counts": dict(video_counts),
        "reliability_quantiles": quantiles(r_values),
        "missing_abnormal_reliability_quantiles": quantiles(missing_abnormal_reliability),
        "target_quantiles": quantiles(y_values),
        "agreement_quantiles": quantiles(agreement_values),
        "positive_ratio_y_gt_0p5": sum(1 for v in y_values if v > 0.5) / max(len(y_values), 1),
        "annotation_override": annotation_stats,
    }
    data["_rali_v2_meta"] = stats
    write_json(Path(args.output), data)
    write_json(Path(args.label_stats), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

