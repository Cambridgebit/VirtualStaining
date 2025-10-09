"""[已弃用] 统一使用 `python -m evaluations.fid_score` 计算 FID/KID。

该脚本仅作为兼容入口，内部调用 evaluations.fid_score 并输出结果。
"""
import argparse
from evaluations.fid_score import (
    calculate_fid_given_paths,
    calculate_kid_given_paths,
)
from typing import Optional
import torch


def _resolve_device(device_str: Optional[str]):
    if device_str in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description='[DEPRECATED] Use python -m evaluations.fid_score')
    parser.add_argument('paths', type=str, nargs=2, help='两个路径：生成集 与 真实集（FID 支持 .npz）')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dims', type=int, default=2048)
    parser.add_argument('--no-kid', action='store_true')
    args = parser.parse_args()

    device = _resolve_device(args.device)
    fid_val = calculate_fid_given_paths(args.paths, batch_size=args.batch_size, device=device, dims=args.dims)
    print(f'[DEPRECATED] Using evaluations.fid_score\nFID: {fid_val:.6f}')
    if not args.no_kid:
        try:
            kid_mean, kid_std = calculate_kid_given_paths(args.paths, batch_size=args.batch_size, device=device, dims=args.dims)
            print(f'KID: {kid_mean:.6f} (+/- {kid_std:.6f})')
        except Exception as e:
            print(f'Skipping KID: {e}')


if __name__ == '__main__':
    main()

