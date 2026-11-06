#!/usr/bin/env python3
"""
Shrink checkpoints for Kaggle upload: keep ONLY the weights used for inference.

Our training checkpoints carry model + optimizer + scheduler + EMA + phys-state
(~150 MB). Figure generation and per-image eval need just the EMA weights, so this
rewrites each to a ~13 MB file with a single 'ema' key (falling back to 'model' /
'params' if a file has no EMA — e.g. DarkIR's released checkpoint).

Usage:
    python strip_ckpt_for_upload.py OUT_DIR CKPT [CKPT ...]

Example (Drive files already copied to ~/Downloads):
    python strip_ckpt_for_upload.py ~/physir-checkpoints \
        ~/Downloads/stage4_physir_final_latest.pt \
        ~/Downloads/stage5_baseline_latest.pt \
        ~/Downloads/stage5_percep_latest.pt \
        ~/Downloads/DarkIR_384.pt

Then zip OUT_DIR and upload it as a private Kaggle Dataset (suggested slug:
<you>/physir-checkpoints). Filenames are preserved, which is what the figure
notebook matches on.
"""
import os
import sys

import torch


def strip(path, out_dir):
    ck = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(ck, dict):
        state = ck.get('ema') or ck.get('model') or ck.get('params')
        which = ('ema' if ck.get('ema') else
                 'model' if ck.get('model') else
                 'params' if ck.get('params') else None)
        epoch = ck.get('epoch')
        if state is None and all(hasattr(v, 'shape') for v in ck.values()):
            state, which, epoch = ck, 'bare state_dict', None
    else:
        state, which, epoch = ck, 'bare state_dict', None

    if state is None:
        raise SystemExit(f'{path}: no weights found under ema/model/params')

    # drop DDP prefixes so the file loads identically single- or multi-GPU
    state = {k.replace('module.', ''): v for k, v in state.items()}

    out = os.path.join(out_dir, os.path.basename(path))
    torch.save({'ema': state, 'source_key': which, 'epoch': epoch}, out)

    mb_in = os.path.getsize(path) / 2**20
    mb_out = os.path.getsize(out) / 2**20
    print(f'{os.path.basename(path):40s} {which:15s} ep={epoch}  '
          f'{mb_in:7.1f} MB → {mb_out:6.1f} MB')


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_dir, ckpts = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    for c in ckpts:
        strip(c, out_dir)
    total = sum(os.path.getsize(os.path.join(out_dir, f))
                for f in os.listdir(out_dir)) / 2**20
    print(f'\n{len(ckpts)} file(s) → {out_dir}  ({total:.1f} MB total) — zip and upload.')


if __name__ == '__main__':
    main()
