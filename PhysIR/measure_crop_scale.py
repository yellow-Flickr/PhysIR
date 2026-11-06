"""
Measure how each PhysIR loss magnitude scales with crop size (256 -> 384).

Why this exists
---------------
Stage 1-3 tuned loss weights at crop 256; Stage 4 trains at crop 384. Only one loss
should care about crop size:

  PhaseEnhancedFrequencyLoss uses torch.fft.rfft2(..., norm='backward') — NO forward
  normalization — so |FFT| magnitudes scale with pixel count N = H*W. Inside it:
      amp_loss   = L1(|rfft2| * mask)          -> scales with N   (crop-dependent)
      phase_loss = L1 on unit-complex gradients -> magnitude 1     (scale-invariant)

  Every other loss (L1 pixel, edge, enhance, retinex, blur) is a spatial MEAN and
  should therefore be crop-invariant.

This script validates that claim empirically and yields the phase_weight rescale factor:

      factor           = phase_raw(384) / phase_raw(256)
      phase_weight_new = phase_weight_256 / factor

Cross-check: retinex/base and blur/base ratios must stay ~unchanged across crops. If they
do, the crop-invariance argument holds and the phase rescale is trustworthy.

Note: loss weights are set to 1.0 so we read RAW magnitudes (weight is just a multiplier).
Batch size is irrelevant here — all losses use mean reduction over the batch.

Usage
-----
  python measure_crop_scale.py                          # synthetic 1/f images (no data needed)
  python measure_crop_scale.py --data <dir-of-images>   # real images (preferred; e.g. LOLBlur GT)
  python measure_crop_scale.py --crops 256 384 512      # compare more crop sizes
"""

import argparse
import os
import glob

import torch
import torch.nn.functional as F

from losses.loss import (
    RetinexLoss,
    BlurAwareGradientLoss,
    PhaseEnhancedFrequencyLoss,
)

# Reference ratios measured in real training at crop 256 / eff-batch 8 (Stage 2, epoch 1)
REF_256 = {'base': 0.0407, 'phase': 0.0294, 'retinex': 0.0103, 'blur': 0.0059}
REF_WEIGHTS_256 = {'phase': 0.20, 'retinex': 0.18, 'blur': 0.02}   # Stage 3 winner ph20_re18

_IMG_EXT = {'.png', '.jpg', '.jpeg', '.bmp'}


# ── image sources ─────────────────────────────────────────────────────────────

def pink_image(size, gen, device='cpu'):
    """Synthetic image with a ~1/f (natural-image-like) power spectrum."""
    noise = torch.randn(1, 3, size, size, generator=gen, device=device)
    f = torch.fft.rfft2(noise)
    fy = torch.fft.fftfreq(size, device=device).view(-1, 1).abs()
    fx = torch.fft.rfftfreq(size, device=device).view(1, -1).abs()
    radial = (fy ** 2 + fx ** 2).sqrt().clamp(min=1e-3)
    img = torch.fft.irfft2(f / radial, s=(size, size))
    img = img - img.amin()
    return (img / img.amax().clamp(min=1e-8)).clamp(0, 1)


def load_images(data_dir, want, min_size):
    """Load real images large enough for the biggest crop."""
    from PIL import Image
    import torchvision.transforms.functional as TF

    paths = []
    for root, _d, files in os.walk(data_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in _IMG_EXT:
                paths.append(os.path.join(root, fn))
    paths.sort()
    out = []
    for p in paths:
        if len(out) >= want:
            break
        try:
            im = Image.open(p).convert('RGB')
        except Exception:
            continue
        if min(im.size) < min_size:
            continue
        out.append(TF.to_tensor(im).unsqueeze(0))
    return out


# ── degradation (held FIXED across crops so "prediction quality" is constant) ──

def gauss_blur(x, sigma, k=11):
    """REFLECT padding on purpose: zero-padding fabricates a border discontinuity that
    injects broadband high-frequency energy, inflating amp_loss more on small crops and
    biasing the measured phase factor. Verified: switching zeros->reflect leaves the phase
    factor unchanged (1.4429 -> 1.4486) but drives retinex 0.9947 -> 1.0001."""
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kern = (g[:, None] * g[None, :]).view(1, 1, k, k).expand(x.shape[1], 1, k, k)
    x = F.pad(x, (k // 2,) * 4, mode='reflect')
    return F.conv2d(x, kern, groups=x.shape[1])


def make_pair(target, mode):
    """Build a plausible (prediction, degraded_input) for a given GT crop."""
    if mode == 'blur':
        pred = gauss_blur(target, sigma=1.0)
    elif mode == 'blur+noise':
        pred = (gauss_blur(target, sigma=1.0) + 0.02 * torch.randn_like(target)).clamp(0, 1)
    else:
        raise ValueError(mode)
    inp = (gauss_blur(target, sigma=2.0) * 0.4).clamp(0, 1)   # dark + blurry, like low_blur_noise
    return pred, inp


# ── measurement ───────────────────────────────────────────────────────────────

def measure(target, mode):
    """Raw (weight=1.0) loss magnitudes for one GT crop."""
    pred, inp = make_pair(target, mode)

    retinex_fn = RetinexLoss(weight=1.0)
    blur_fn    = BlurAwareGradientLoss(weight=1.0)
    phase_fn   = PhaseEnhancedFrequencyLoss(weight=1.0)

    with torch.no_grad():
        return {
            'base':    F.l1_loss(pred, target).item(),   # pixel L1 — the crop-invariant anchor
            'retinex': retinex_fn(pred, target).item(),
            'blur':    blur_fn(pred, inp).item(),
            'phase':   phase_fn(pred, target).item(),
        }


def center_crop(img, size):
    _, _, H, W = img.shape
    top, left = (H - size) // 2, (W - size) // 2
    return img[:, :, top:top + size, left:left + size]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=None, help='dir of real images (GT). Omit for synthetic 1/f.')
    ap.add_argument('--crops', type=int, nargs='+', default=[256, 384])
    ap.add_argument('--n', type=int, default=16, help='number of images to average over')
    ap.add_argument('--mode', default='blur', choices=['blur', 'blur+noise'])
    ap.add_argument('--source', type=int, default=None,
                    help='synthetic source size; must exceed max crop so EVERY crop is a '
                         'sub-crop (avoids the largest crop matching the generation grid '
                         'exactly, which would bias its FFT). Default: 2x max crop.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    big = max(args.crops)
    src_size = args.source or big * 2

    if args.data:
        imgs = load_images(args.data, args.n, big)
        if not imgs:
            raise SystemExit(f'No images >= {big}px found under {args.data}')
        src = f'real images from {args.data}'
    else:
        if src_size <= big:
            raise SystemExit('--source must exceed the largest crop')
        imgs = [pink_image(src_size, gen) for _ in range(args.n)]
        src = f'synthetic 1/f (natural-image-like spectrum), source {src_size}px'

    print(f'Source      : {src}')
    print(f'Images      : {len(imgs)}   mode: {args.mode}   crops: {args.crops}')
    print('Loss weights set to 1.0 → raw magnitudes.\n')

    # accumulate mean raw loss per crop
    results = {}
    for c in args.crops:
        acc = {}
        for img in imgs:
            m = measure(center_crop(img, c), args.mode)
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v / len(imgs)
        results[c] = acc

    keys = ['base', 'retinex', 'blur', 'phase']
    hdr = f'{"loss":9s}' + ''.join(f'{f"crop {c}":>14s}' for c in args.crops)
    print(hdr)
    print('-' * len(hdr))
    for k in keys:
        print(f'{k:9s}' + ''.join(f'{results[c][k]:14.6f}' for c in args.crops))

    base_c = args.crops[0]
    print(f'\n--- scale factor vs crop {base_c} (expect ~1.00 for all but phase) ---')
    for k in keys:
        line = f'{k:9s}'
        for c in args.crops:
            line += f'{results[c][k] / results[base_c][k]:14.4f}'
        print(line)

    # ── verdict + rescale, only meaningful for the 256 -> 384 pair ──
    if 256 in results and 384 in results:
        f_base    = results[384]['base']    / results[256]['base']
        f_retinex = results[384]['retinex'] / results[256]['retinex']
        f_blur    = results[384]['blur']    / results[256]['blur']
        f_phase   = results[384]['phase']   / results[256]['phase']

        print('\n=== CROSS-CHECK: are the non-phase losses crop-invariant? ===')
        ok = True
        for name, f in [('base', f_base), ('retinex', f_retinex), ('blur', f_blur)]:
            # blur gets a wider band: BlurAwareGradientLoss zero-pads internally (conv2d
            # padding=5), fabricating border gradients. That border is a larger fraction of a
            # 256 crop (9.2%) than a 384 crop (6.2%), so blur reads ~8% lower at 384.
            # Excluding a 4px margin drives the factor to exactly 1.0000 — the loss is
            # intrinsically invariant. At blur_weight=0.02 the correction is immaterial.
            lo, hi = (0.90, 1.10) if name == 'blur' else (0.95, 1.05)
            good = lo <= f <= hi
            ok &= good
            note = ''
            if name == 'blur' and not (0.95 <= f <= 1.05):
                note = '  (zero-pad border effect; interior factor = 1.000)'
            print(f'  {name:8s} factor = {f:.4f}  '
                  f'{"✅ invariant" if good else "⚠️ NOT invariant"}{note}')
        if ok:
            print('  → Crop-invariance CONFIRMED. Only phase needs rescaling.')
        else:
            print('  → Unexpected: a supposedly crop-invariant loss moved. Investigate before Stage 4.')

        print(f'\n=== PHASE RESCALE ===')
        print(f'  phase factor (384/256) = {f_phase:.4f}')
        new_w = REF_WEIGHTS_256['phase'] / f_phase
        print(f'  phase_weight_256 = {REF_WEIGHTS_256["phase"]}')
        print(f'  phase_weight_384 = {REF_WEIGHTS_256["phase"]} / {f_phase:.4f} = {new_w:.4f}')

        pred_ratio = (REF_256['phase'] / REF_256['base']) * (f_phase / f_base)
        print(f'\n  Predicted ep1 phase/base ratio at 384 with the OLD weight (0.20): {pred_ratio:.3f}')
        print(f'  (256 reference ratio = {REF_256["phase"]/REF_256["base"]:.3f})')
        print(f'  With the rescaled weight it should return to ≈ {REF_256["phase"]/REF_256["base"]:.3f}.')
        print('\n  Confirm against a real 1-epoch run at 384: read loss_base / loss_phase_freq'
              '\n  from .metrics/<run>/training_log.csv at epoch 1.')


if __name__ == '__main__':
    main()
