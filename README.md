# RALI-VAD

Official code repository for **RALI-VAD: Reliability-Aware Weakly Supervised
Video Anomaly Detection with Training-Only Multimodal Semantic Supervision**.

RALI-VAD combines two imperfect supervision sources during training:

- dense visual pseudo scores produced by a weakly supervised visual detector;
- offline semantic evidence produced by a frozen multimodal large language model.

Reliability-Aware Label Integration (RALI) constructs soft targets and
reliability weights from target-extremity proxies, source agreement, and
semantic coverage. Noise-Adaptive Refinement (NAR) then trains the visual
detector with reliability-weighted classification, entropy sharpening, and a
momentum target bank. Inference uses only the visual detector.

## Repository structure

| File | Purpose |
|---|---|
| `main.py`, `train.py` | Stage 1 weakly supervised training |
| `model.py`, `layers.py` | SLTR visual detector and relation layers |
| `generate_pseudo_generic.py` | Dense visual pseudo-score generation |
| `rali_v2.py` | RALI target and reliability construction |
| `prepare_rali_v2_dataset.py` | Stage 2 manifest construction |
| `main_rali_v2.py`, `train_rali_v2.py` | NAR Stage 2 training |
| `dataset.py`, `dataset_rali_v2.py` | Feature-manifest datasets |
| `test.py`, `utils.py` | Evaluation and sequence utilities |
| `configs/paper_config.yaml` | Final paper configuration |

## Requirements

```bash
pip install -r requirements.txt
```

The code uses pre-extracted CLIP (768-dimensional) and I3D
(2,048-dimensional) clip features. Dataset videos and extracted features are
not redistributed; obtain UCF-Crime, XD-Violence, and UBnormal from their
official providers and follow their respective terms.

## Core workflow

### 1. Train the Stage 1 visual detector

```bash
python main.py \
  --dataset <stage1_manifest.json> \
  --gt_path <frame_labels.npy> \
  --output_path <stage1_output> \
  --training_name stage1_seed0 \
  --batch-size 4 \
  --max_epoch 20 \
  --seed 0
```

### 2. Generate visual pseudo scores

```bash
python generate_pseudo_generic.py --help
```

### 3. Construct RALI targets

```bash
python prepare_rali_v2_dataset.py \
  --base-pseudo <visual_pseudo.json> \
  --noise-label <semantic_records.json> \
  --output <stage2_manifest.json> \
  --label-stats <rali_statistics.json> \
  --alpha 0.65 \
  --tau 0.20 \
  --r-min 0.05
```

### 4. Train the NAR detector

```bash
python main_rali_v2.py \
  --stage2_dataset <stage2_manifest.json> \
  --init_checkpoint <stage1_checkpoint.pth> \
  --output_path <stage2_output> \
  --training_name rali_vad_seed0 \
  --batch-size 4 \
  --max_epoch 40 \
  --warmup_epoch 5 \
  --use_label_update \
  --use_sharpness_loss \
  --lambda_s 0.5 \
  --rali_beta 0.9 \
  --seed 0
```

The formal seeds are `0`, `1`, and `2`. The final UCF-Crime seed-0
configuration reports 89.59% AUC and 42.03% AP.

## Final configuration

| Component | Value |
|---|---:|
| Visual input | CLIP 768 + I3D 2048 |
| Sequence length | 256 |
| Local attention | 2 layers, 1 head, window 8 |
| Stage 1 epochs | 20 |
| Stage 2 epochs | 40 |
| RALI visual prior `alpha` | 0.65 |
| Agreement threshold `tau` | 0.20 |
| Reliability floor `r_min` | 0.05 |
| Target momentum `beta` | 0.90 |
| Sharpening weight `lambda_s` | 0.5 |
| Target-update warm-up | 5 epochs |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-2 |

## Data and licenses

See [DATA.md](DATA.md) for dataset handling. Code is released under the
[MIT License](LICENSE).

## Citation

Please cite the accompanying manuscript:

> Hanwei Li, Ke Xu, Xinghao Jiang, and Tanfeng Sun. RALI-VAD: Reliability-Aware Weakly
> Supervised Video Anomaly Detection with Training-Only Multimodal Semantic
> Supervision. Pattern Recognition, manuscript submitted.

## Contact

Xinghao Jiang (corresponding author), Shanghai Jiao Tong University.\nEmail: xhjiang@sjtu.edu.cn.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China
(grant numbers 62372295 and 62272299) and the Shanghai Municipal Science and
Technology Major Project.
