"""
Paired significance test for the PhysIR ablation table (methodology #12).

Compares two configs' per-image test metrics with the Wilcoxon signed-rank test — a formal
significance test that works from a SINGLE training run per config, because it tests whether the
per-image improvement is *systematic* across the ~1800 test images (not whether two run-means
differ). Non-parametric → robust to the skewed PSNR/LPIPS distributions.

Reports, per metric: p-value, median improvement, and rank-biserial effect size.

Inputs are the `per_image_<config>.csv` files (columns: idx, psnr, ssim, lpips) written by the
Stage 4 / Stage 5 eval passes. Rows are aligned by `idx` (deterministic test-loader order), so
row i is the same image in both files.

Usage:
    python paired_wilcoxon.py per_image_abl_full.csv per_image_abl_baseline.csv
    python paired_wilcoxon.py <A.csv> <B.csv> --name-a full --name-b baseline
"""

import argparse
import csv
import math


def load(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[int(r['idx'])] = {k: float(r[k]) for k in ('psnr', 'ssim', 'lpips')}
    return rows


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank on paired differences (a - b), zeros dropped.
    Normal approximation with tie + continuity correction. Returns (W, z, p, n_nonzero)."""
    d = [x for x in diffs if x != 0.0]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # average rank for ties (1-indexed)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    W_plus = sum(ranks[i] for i in range(n) if d[i] > 0)
    W_minus = sum(ranks[i] for i in range(n) if d[i] < 0)
    W = min(W_plus, W_minus)
    mean_W = n * (n + 1) / 4.0
    # tie correction for variance
    from collections import Counter
    tie = Counter(abs(x) for x in d)
    tie_term = sum(t ** 3 - t for t in tie.values())
    var_W = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_W <= 0:
        return W, 0.0, 1.0, n
    z = (W - mean_W + 0.5 * (1 if W < mean_W else -1)) / math.sqrt(var_W)  # continuity-corrected
    p = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))  # two-sided normal tail
    return W, z, min(1.0, p), n


def median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv_a'); ap.add_argument('csv_b')
    ap.add_argument('--name-a', default='A'); ap.add_argument('--name-b', default='B')
    args = ap.parse_args()

    A, B = load(args.csv_a), load(args.csv_b)
    idx = sorted(set(A) & set(B))
    if len(idx) != len(A) or len(idx) != len(B):
        print(f'⚠️  {len(A)} vs {len(B)} rows; using {len(idx)} shared idx.')

    # higher-is-better for psnr/ssim; lower-is-better for lpips → flip sign so "positive = A better"
    sign = {'psnr': 1.0, 'ssim': 1.0, 'lpips': -1.0}
    print(f'\nPaired Wilcoxon signed-rank:  {args.name_a}  vs  {args.name_b}   (n={len(idx)})')
    print(f'  "improvement" = {args.name_a} − {args.name_b} (LPIPS sign-flipped so + = better)\n')
    print(f'{"metric":7} {"median Δ":>12} {"% A better":>11} {"z":>8} {"p-value":>11} {"effect r":>9} sig')
    print('-' * 64)
    for m in ('psnr', 'ssim', 'lpips'):
        diffs = [sign[m] * (A[i][m] - B[i][m]) for i in idx]
        med = median(diffs)
        frac = 100.0 * sum(1 for x in diffs if x > 0) / len(diffs)
        W, z, p, n = wilcoxon_signed_rank(diffs)
        r = abs(z) / math.sqrt(n) if n else 0.0   # rank-biserial-ish effect size
        stars = '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else 'ns'
        raw_med = median([A[i][m] - B[i][m] for i in idx])   # report in native units
        print(f'{m:7} {raw_med:+12.4f} {frac:10.1f}% {z:8.2f} {p:11.2e} {r:9.3f} {stars}')
    print('\n  *** p<0.001  ** p<0.01  * p<0.05  ns = not significant')
    print('  effect r: ~0.1 small, ~0.3 medium, ~0.5 large')


if __name__ == '__main__':
    main()
