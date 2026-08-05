import argparse

parser = argparse.ArgumentParser(description='MMCT')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--output_path', default='./output/', type=str)
parser.add_argument('--training_name', default='svad_LNL_onlywarmup_onlynoise', type=str)

parser.add_argument('--embed_dim', default=512, type=int)
parser.add_argument('--visual_length', default=256, type=int)
parser.add_argument('--visual_head', default=1, type=int)
parser.add_argument('--visual_layers', default=2, type=int)
parser.add_argument('--attn_window', default=8, type=int)

parser.add_argument('--max_epoch', default=40, type=int)
parser.add_argument('--checkpoint_path', default='', type=str)
parser.add_argument('--use_checkpoint', default=False, type=bool)
parser.add_argument('--init_checkpoint', default='', type=str)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--dataset', default='./dataset/ucf_crime.json')
parser.add_argument('--stage2_dataset', default='', type=str)
parser.add_argument('--gt_path', default='./dataset/gt_ucf.npy')

parser.add_argument('--lr', default=1e-4)
parser.add_argument('--scheduler_rate', default=0.1)
parser.add_argument('--scheduler_milestones', default=[7, 14])

parser.add_argument('--p_threshold', default=0.5, type=float)
parser.add_argument('--warmup_epoch', default=5, type=int)
parser.add_argument('--feature_mode', default='clip_i3d', choices=['clip', 'i3d', 'clip_i3d'])
parser.add_argument('--use_label_update', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--use_sharpness_loss', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--lambda_s', default=0.5, type=float)
parser.add_argument('--rali_beta', default=0.9, type=float)
parser.add_argument('--max_steps', default=0, type=int)
parser.add_argument('--eval_interval', default=0, type=int)

