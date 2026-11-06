"""
Clip-level train/val split for the LOLBlur dataset.

Must run ONCE before Stage 1 pilots. Splits at the video-clip level so no
two frames from the same clip appear in both train and val (no temporal leak).

Usage:
    python split_lolblur.py

Output:
    data/datasets/LOLBlur/physir_train/low_blur_noise/   (~90% of clips)
    data/datasets/LOLBlur/physir_train/high_sharp_scaled/
    data/datasets/LOLBlur/physir_val/low_blur_noise/     (~10% of clips)
    data/datasets/LOLBlur/physir_val/high_sharp_scaled/

Uses symlinks — no data is copied. Re-running is safe (existing links are skipped).
"""

import os
import random
from pathlib import Path

SEED     = 42
VAL_FRAC = 0.10

SRC  = Path('data/datasets/LOLBlur/train')
DEST = Path('data/datasets/LOLBlur')

LOW_SUB  = 'low_blur_noise'
HIGH_SUB = 'high_sharp_scaled'

random.seed(SEED)

src_low  = SRC / LOW_SUB
src_high = SRC / HIGH_SUB

if not src_low.exists():
    raise FileNotFoundError(
        f'{src_low} not found. Place LOLBlur/train/ at data/datasets/LOLBlur/train/')

clips = sorted(p.name for p in src_low.iterdir() if p.is_dir())
if not clips:
    raise ValueError(f'No clip subdirectories found in {src_low}')

random.shuffle(clips)
n_val   = max(1, int(len(clips) * VAL_FRAC))
val_set = set(clips[:n_val])

print(f'Total clips:  {len(clips)}')
print(f'Val clips:    {n_val}  ({n_val/len(clips)*100:.1f}%)')
print(f'Train clips:  {len(clips) - n_val}  ({(len(clips)-n_val)/len(clips)*100:.1f}%)')
print()

splits = {
    'physir_train': [c for c in clips if c not in val_set],
    'physir_val':   [c for c in clips if c in val_set],
}

for split_name, split_clips in splits.items():
    for sub in (LOW_SUB, HIGH_SUB):
        src_sub = SRC / sub
        dst_sub = DEST / split_name / sub
        dst_sub.mkdir(parents=True, exist_ok=True)

        for clip in split_clips:
            src_clip = src_sub / clip
            dst_clip = dst_sub / clip
            if dst_clip.exists() or dst_clip.is_symlink():
                continue  # skip if already linked
            os.symlink(src_clip.resolve(), dst_clip)

    n_imgs = sum(
        len(list((DEST / split_name / LOW_SUB / c).iterdir()))
        for c in split_clips
    )
    print(f'{split_name}: {len(split_clips)} clips, ~{n_imgs} images')

print('\nDone. Update pilot YML configs to use physir_train/ and physir_val/.')
print('Val clips are deterministic (seed=42) — re-running produces the same split.')
