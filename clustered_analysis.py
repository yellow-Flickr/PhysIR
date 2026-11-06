#!/usr/bin/env python3
"""
Cluster-aware paired analysis for the PhysIR comparisons.

WHY THIS EXISTS
---------------
The LOLBlur test set is 30 clips x 60 temporally adjacent frames. Frames inside a clip are
near-duplicates, so the 1800 per-image records are NOT 1800 independent observations — there
are effectively ~30. Treating them as independent (a plain per-image Wilcoxon over n=1800) is
pseudo-replication and inflates significance by a large factor.

This script reports, for each comparison and metric:
  * frame-level Wilcoxon (n=1800)          — retained for comparison, NOT the headline
  * clip-level Wilcoxon (n=30 clip means)  — cluster-aware, the defensible test
  * clip cluster bootstrap 95% CI          — the most informative summary
  * clips favouring A, and the variance split (between- vs within-clip)

Clip membership is inferred from row order: the dataset sorts full paths, so rows arrive in
clip-major order and clip = idx // frames_per_clip. This is verified against the `fname`
column (filenames ascend within a clip and reset at each boundary) and the script refuses to
run if the structure does not check out.

Usage
-----
    python clustered_analysis.py /path/to/per_image_corrected            # full suite
    python clustered_analysis.py DIR --a per_image_x.csv --b per_image_y.csv --label "x vs y"
    python clustered_analysis.py DIR --md out.md                        # markdown table
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

METRICS = ('psnr', 'ssim', 'lpips', 'dists')
LOWER_BETTER = ('lpips', 'dists')  # sign-flipped so "+" always means "A is better"

# (label, file A stem, file B stem) — A is the arm expected to be better
SUITE = [
    ('full vs baseline @384',        'stage4_physir_final',   'stage5_baseline'),
    ('full vs percep @384',          'stage4_physir_final',   'stage5_percep'),
    ('percep vs baseline @384',      'stage5_percep',         'stage5_baseline'),
    ('PhysIR vs DarkIR-m released',  'physir_full',           'darkir_m_released'),
    ('percep-repro vs DarkIR-m',     'abl_percep384',         'darkir_m_released'),
    ('full vs baseline @256',        'abl_full_last',         'abl_baseline_last'),
    ('phase vs baseline @256',       'abl_phase_last',        'abl_baseline_last'),
    ('retinex vs baseline @256',     'abl_retinex_last',      'abl_baseline_last'),
    ('percep vs baseline @256',      'abl_percep_last',       'abl_baseline_last'),
    ('blur vs baseline @256',        'abl_blur_last',         'abl_baseline_last'),
    ('full vs phase @256',           'abl_full_last',         'abl_phase_last'),
    ('full vs retinex @256',         'abl_full_last',         'abl_retinex_last'),
    ('full vs percep @256',          'abl_full_last',         'abl_percep_last'),
    ('full+percep vs full @256',     'abl_full_percep_last',  'abl_full_last'),
    ('blur-noedge vs edge+blur',     'abl_blurnoedge_last',   'abl_blur_last'),
]


def load(path, dists_dir=None, stem=None):
    rows = list(csv.DictReader(open(path)))
    if 'fname' not in rows[0]:
        raise SystemExit(
            f'{path} has no `fname` column — this is a LEGACY per-image CSV written before the '
            'shuffle=False fix. Its row order is not a stable image id, so every paired test '
            'built on it is invalid. Use the per_image_corrected/ set.')
    data = {m: np.array([float(r[m]) for r in rows]) for m in METRICS if m in rows[0]}
    names = [r['fname'] for r in rows]

    # DISTS lives in its own file (per_image_dists_<stem>.csv, cols idx,fname,dists) because it
    # was computed in a later pass. Merge it on idx, but only after proving the fname sequences
    # agree — an unverified join here would reintroduce exactly the bug the fname column exists
    # to catch.
    if dists_dir and stem:
        dp = os.path.join(dists_dir, f'per_image_dists_{stem}.csv')
        if os.path.exists(dp):
            drows = list(csv.DictReader(open(dp)))
            dnames = [r['fname'] for r in drows]
            # The two passes label rows differently: the DISTS pass writes clip-qualified names
            # ("0012/0010.png"), the earlier pass wrote bare basenames ("0010.png"). Bare
            # basenames are NOT unique — 90 distinct values across 1800 rows — so compare on the
            # normalised key positionally, and separately require that the clip-qualified side
            # IS unique. Together that pins a genuine 1:1 row correspondence; matching bare
            # names alone would not.
            if len(dnames) != len(names):
                raise SystemExit(f'{dp}: {len(dnames)} rows vs {len(names)} in {path}')
            if len(set(dnames)) != len(dnames):
                raise SystemExit(f'{dp}: fname values are not unique — cannot verify the join')
            if [os.path.basename(x) for x in dnames] != [os.path.basename(x) for x in names]:
                raise SystemExit(f'{dp}: row order differs from {path} — refusing to join')
            data['dists'] = np.array([float(r['dists']) for r in drows])
    return data, names


def infer_clips(names, n):
    """Clip boundaries from filename resets; falls back to 60 frames/clip."""
    if not names:
        print('  ! no fname column — assuming 60 frames per clip (unverified)', file=sys.stderr)
        return n // 60
    bounds = [0] + [i for i in range(1, n) if names[i] < names[i - 1]]
    sizes = [(bounds[i + 1] if i + 1 < len(bounds) else n) - bounds[i]
             for i in range(len(bounds))]
    if len(set(sizes)) != 1:
        raise SystemExit(f'clip sizes are not uniform: {sorted(set(sizes))} — inspect the data')
    return len(bounds)


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p via the normal approximation with tie correction."""
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    srt = np.abs(d)[order]
    i = 0
    while i < n:                                   # average tied ranks
        j = i
        while j + 1 < n and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    W = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu) / sd if sd else 0.0
    return math.erfc(abs(z) / math.sqrt(2))


def stars(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'


def analyse(dA, dB, metric, n_clips, boots, rng):
    s = -1.0 if metric in LOWER_BETTER else 1.0
    d = s * (dA[metric] - dB[metric])
    frame_p = wilcoxon_p(d)
    per_clip = d.reshape(n_clips, -1)
    clip_d = per_clip.mean(1)
    clip_p = wilcoxon_p(clip_d)
    idx = rng.integers(0, n_clips, (boots, n_clips))
    boot = clip_d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(clip_d.mean()), lo=float(lo), hi=float(hi),
                frame_p=frame_p, clip_p=clip_p,
                pos=int((clip_d > 0).sum()), n_clips=n_clips,
                between=float(clip_d.std(ddof=1)), within=float(per_clip.std(1, ddof=1).mean()),
                sig=lo * hi > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--a'), ap.add_argument('--b'), ap.add_argument('--label', default='A vs B')
    ap.add_argument('--boots', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--md', help='also write a markdown table here')
    ap.add_argument('--dists', help='dir of per_image_dists_<name>.csv to merge on fname')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pairs = ([(args.label, args.a.replace('per_image_', '').replace('.csv', ''),
               args.b.replace('per_image_', '').replace('.csv', ''))]
             if args.a and args.b else SUITE)

    cache, n_clips = {}, None
    def get(stem):
        nonlocal n_clips
        if stem not in cache:
            p = os.path.join(args.dir, f'per_image_{stem}.csv')
            if not os.path.exists(p):
                return None
            d, names = load(p, args.dists, stem)
            k = infer_clips(names, len(next(iter(d.values()))))
            n_clips = k if n_clips is None else n_clips
            if k != n_clips:
                raise SystemExit(f'{stem}: {k} clips, expected {n_clips}')
            cache[stem] = d
        return cache[stem]

    rows, md = [], []
    print(f'Cluster-aware paired analysis — clip-level is the headline; '
          f'frame-level shown for contrast\n{"="*104}')
    hdr = f'{"comparison":28s} {"metric":6s} {"clip mean [95% CI]":>26s} {"clip p":>9s} {"frame p":>10s} {"clips+":>7s}'
    print(hdr); print('-' * 104)
    for label, a, b in pairs:
        dA, dB = get(a), get(b)
        if dA is None or dB is None:
            print(f'{label:28s} (missing input — skipped)'); continue
        for m in METRICS:
            if m not in dA or m not in dB:
                continue
            r = analyse(dA, dB, m, n_clips, args.boots, rng)
            ci = f"{r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"
            flag = '✅' if r['sig'] else '  '
            print(f'{label:28s} {m:6s} {ci:>26s}{flag} {stars(r["clip_p"]):>7s} '
                  f'{stars(r["frame_p"]):>10s} {r["pos"]:>4d}/{r["n_clips"]}')
            md.append(f"| {label} | {m} | {r['mean']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                      f"{stars(r['clip_p'])} | {stars(r['frame_p'])} | {r['pos']}/{r['n_clips']} |")
            rows.append(r)
        print('-' * 104)

    ex = rows[0]
    print(f'\nn = {ex["n_clips"]} clips x {1800 // ex["n_clips"]} frames. '
          f'Between-clip sd {ex["between"]:.3f} vs within-clip sd {ex["within"]:.3f} '
          f'(first comparison) — clip identity dominates, which is why the frame-level test '
          f'overstates the evidence.')
    print('✅ = clip-level 95% bootstrap CI excludes zero. LPIPS sign-flipped so + = A better.')

    if args.md:
        with open(args.md, 'w') as f:
            f.write('| comparison | metric | clip mean Δ | 95% CI (clip bootstrap) | '
                    'clip-level p | frame-level p | clips favouring A |\n')
            f.write('|---|---|---|---|---|---|---|\n')
            f.write('\n'.join(md) + '\n')
        print('markdown →', args.md)


if __name__ == '__main__':
    main()
