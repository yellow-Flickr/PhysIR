#!/usr/bin/env python3
"""
Local per-image test-set eval (CPU/MPS) — produces the per_image_<name>.csv needed by
paired_wilcoxon.py, identical in format and idx-order to the Colab notebooks' eval loop.

Usage:
  python eval_per_image_local.py \
      --ckpt /path/to/stage4_latest.pt \
      --test "/Volumes/NEW VOLUME/Datasets/LOLBlur/test" \
      --out per_image_stage4_full.csv

Values are device-independent (inference only); MPS is auto-picked on Apple Silicon for
~10x speed over CPU. idx alignment matches the Colab runs because the same create_test_data
reader is used (batch 1, sorted order, no shuffle).
"""
import os, csv, argparse, time

import numpy as np
import torch

import train as T                       # build_model, _ssim_metric, _LPIPS
from data import create_test_data

NET_CFG = {'name': 'DarkIR', 'img_channels': 3, 'width': 32,
           'middle_blk_num_enc': 2, 'middle_blk_num_dec': 2,
           'enc_blk_nums': [1, 2, 3], 'dec_blk_nums': [3, 1, 1],
           'dilations': [1, 4, 9], 'extra_depth_wise': True}


def pick_device(arg):
    if arg != 'auto':
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint .pt (EMA weights used if present)')
    ap.add_argument('--test', required=True, help='LOLBlur test dir (contains low_blur_noise/ + high_sharp_scaled/)')
    ap.add_argument('--out', required=True, help='output per-image CSV path')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    ap.add_argument('--raw', action='store_true', help='use raw (non-EMA) weights even if EMA present')
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f'device: {device}')

    assert T._LPIPS_AVAILABLE, 'lpips not installed — pip install lpips'
    lpips_fn = T._LPIPS(net='vgg', verbose=False).to(device)

    opt_test = {'name': 'LOLBlur',
                'val': {'test_path': args.test, 'batch_size_test': 1},
                'train': {'n_workers': 2, 'verbose': True}}
    tl_raw, _ = create_test_data(rank=0, world_size=1, opt=opt_test)
    tl = tl_raw if not isinstance(tl_raw, dict) else list(tl_raw.values())[0]['loader']

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if not isinstance(ck, dict) or all(hasattr(v, 'shape') for v in ck.values()):
        state, which = ck, 'raw state_dict'          # bare state_dict file
    else:
        # ours: ema -> model; DarkIR release: 'params'
        state = (ck.get('ema') if not args.raw else None) or ck.get('model') or ck.get('params')
        which = 'ema' if (ck.get('ema') and not args.raw) else \
                ('model' if ck.get('model') else 'params')
    assert state is not None, 'no weights found under ema/model/params'
    print(f"checkpoint: epoch {ck.get('epoch') if isinstance(ck, dict) else '?'} · weights: {which}")

    model = T.build_model(NET_CFG).to(device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)
    model.eval()
    T._ssim_metric.to(device)

    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, (high, low) in enumerate(tl):
            high, low = high.to(device), low.to(device)
            pred = model(low, side_loss=False).clamp(0, 1)
            _, _, H, W = high.shape
            pred = pred[:, :, :H, :W]
            mse = torch.mean((pred - high) ** 2).item()
            psnr = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
            ssim = T._ssim_metric(pred, high).item()
            lp = lpips_fn(pred * 2 - 1, high * 2 - 1).mean().item()   # correct [-1,1] (primary)
            # DarkIR's release evaluates LPIPS on unscaled [0,1] inputs (test_utils.py L57) —
            # emit that convention too, solely for cross-paper comparability columns.
            lp_dk = lpips_fn(pred, high).mean().item()
            rows.append((i, psnr, ssim, lp, lp_dk))
            if (i + 1) % 100 == 0:
                rate = (time.time() - t0) / (i + 1)
                print(f'  {i+1} imgs · {rate:.2f} s/img · ~{rate*(len(tl)-i-1)/60:.0f} min left')

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'psnr', 'ssim', 'lpips', 'lpips_darkirconv'])
        w.writerows(rows)

    a = np.array([r[1:] for r in rows])
    print(f'\nwrote {args.out}  (n={len(a)})')
    print(f'means: PSNR {a[:,0].mean():.4f} · SSIM {a[:,1].mean():.4f} · '
          f'LPIPS {a[:,2].mean():.4f} · LPIPS(DarkIR-conv) {a[:,3].mean():.4f}')
    print('cross-check PSNR/SSIM against known references before running Wilcoxon.')


if __name__ == '__main__':
    main()
