# Smoke-Testing the Training Loop

Before committing to a 300-epoch run on a full dataset, run a short end-to-end smoke test
to confirm the training loop, all four phases, checkpointing, and (optionally) the val
eval all cycle without errors. This should take under 5 minutes.

---

## Option A — Use images already in the repo (zero download)

The repo contains 4 paired images in `assets/qualis/` (1120×640 px each):

```
assets/qualis/inputs/    ← low-light images  (0010, 0075, 0087, 0088)
assets/qualis/results/   ← ground-truth images (same filenames)
```

These are enough to verify the full loop. They are NOT representative training data —
the model will overfit immediately and PSNR numbers mean nothing — but that is fine.
You are only checking that the code runs.

---

## Option B — Download a small public dataset (~30 MB, 100 pairs)

LOLv2-real has a flat directory structure and 100 test pairs, making it the easiest
real dataset to verify with. It mirrors what you will use for evaluation later.

```bash
# Install gdown if needed
pip install gdown

# LOLv2 (real + synthetic, ~120 MB total)
gdown --fuzzy "https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view" -O LOLv2.zip
unzip LOLv2.zip -d data/datasets/
```

Expected layout after unzip:
```
data/datasets/LOL-v2/
├── Real_captured/
│   ├── train/
│   │   ├── Low/      ← 689 images
│   │   └── Normal/   ← 689 images
│   └── test/
│       ├── Low/      ← 100 images
│       └── Normal/   ← 100 images
└── Synthetic/
    ├── train/
    │   ├── Low/      ← 900 images
    │   └── Normal/
    └── test/
```

Use `LOL-v2/Real_captured/train/Low` and `Normal` as your `low_path` / `high_path`
for the smoke test. 100 pairs from the test split also work if you want a quicker
download — just note the train/test contamination does not matter for a loop check.

---

## Setting up the smoke-test config

Create `options/train/smoke_test.yml` (copy `baseline.yml` and apply these changes):

```yaml
datasets:
  train:
    # Option A — repo assets (4 pairs)
    low_path:  ./assets/qualis/inputs
    high_path: ./assets/qualis/results

    # Option B — LOLv2-real train split (689 pairs)
    # low_path:  ./data/datasets/LOL-v2/Real_captured/train/Low
    # high_path: ./data/datasets/LOL-v2/Real_captured/train/Normal

    crop_size: 128    # smaller patches = faster iterations
    batch_size: 1     # 1 keeps memory low; use 2 if you have headroom
    n_workers: 0      # 0 avoids multiprocessing issues during debugging
    use_flips: True
  val:
    test_path: ./data/datasets    # leave as-is; val skipped if path is missing
    batch_size_test: 1
    verbose: False
    n_workers: 0

network:
  name: DarkIR
  img_channels: 3
  width: 32
  middle_blk_num_enc: 2
  middle_blk_num_dec: 2
  enc_blk_nums: [1, 2, 3]
  dec_blk_nums: [3, 1, 1]
  dilations: [1, 4, 9]
  extra_depth_wise: True

save:
  pretrained: null
  path: ./models/smoke_latest.pt
  best: ./models/bests/smoke_best.pt
  save_freq: 2      # checkpoint every 2 epochs
  eval_freq: 999    # disable val eval (set high so it never triggers)

pixel_criterion: l1
perceptual: False
edge: True
edge_weight: 0.05
edge_criterion: l2
edge_reduction: mean
frequency: False
enhance: False       # EnhanceLoss uses VGG — skip for a fast smoke test

physics_losses:
  enabled: True
  phase2_start: 3   # compress the schedule so all 4 phases are tested in 10 epochs
  phase3_start: 5
  phase4_start: 7
  retinex: True
  retinex_weight: 0.10
  blur: True
  blur_weight: 0.05
  phase: True
  phase_weight: 0.05

train:
  epochs: 10        # 10 epochs covers all four phases with the compressed schedule
  lr_initial: 0.0002
  weight_decay: 0.00001
  betas: [0.9, 0.9]
  lr_scheme: CosineAnnealing
  eta_min: 0.000001
  grad_clip: 1.0

wandb:
  init: False

Resize: False
```

Key differences from the real configs:
- `epochs: 10` with compressed phase boundaries (3 / 5 / 7) so all four phases
  activate within the 10-epoch run
- `enhance: False` to skip the VGG-based EnhanceLoss (saves a torchvision download
  and ~500 MB of GPU memory)
- `n_workers: 0` to surface any multiprocessing errors as clean stack traces
- `eval_freq: 999` disables validation (your val dataset probably is not set up yet)

---

## Run it

```bash
python train.py -p options/train/smoke_test.yml
```

Expected console output, epoch by epoch:

```
Using device: mps          ← or cuda / cpu depending on your machine

── Phase 1 (epoch 1) ── L1 + Edge
Epoch   1 | total=0.1234  base=0.1234  retinex=0.0000  blur=0.0000  phase=0.0000  lr=2.00e-04

Epoch   2 | total=0.1189  base=0.1189  ...

── Phase 2 (epoch 3) ── L1 + Edge + Retinex
Epoch   3 | total=0.1301  base=0.1150  retinex=0.0151  blur=0.0000  phase=0.0000  ...

── Phase 3 (epoch 5) ── L1 + Edge + Retinex + BlurGrad
Epoch   5 | total=0.1380  base=0.1140  retinex=0.0148  blur=0.0092  phase=0.0000  ...

── Phase 4 (epoch 7) ── L1 + Edge + Retinex + BlurGrad + PhaseFreq  [FULL]
Epoch   7 | total=0.1450  base=0.1130  retinex=0.0145  blur=0.0090  phase=0.0085  ...
  Checkpoint saved (epoch 8)

Epoch  10 | ...
Training complete.
```

---

## What to check

### The loop is working if:

| Signal | What it means |
|---|---|
| Phase banners print at epochs 1, 3, 5, 7 | Progressive schedule is wired up correctly |
| `retinex`, `blur`, `phase` columns are 0.0000 in Phase 1 and non-zero once their phase activates | Physics losses are gating correctly |
| `base` loss decreases across epochs | Gradient flow through the model is healthy |
| `blur_sigma` column appears and changes (slightly) from epoch to epoch | Learnable PSF sigma is receiving gradients |
| Checkpoint files appear in `models/` | Saving works |
| No CUDA / MPS out-of-memory crash | Memory budget is OK at `batch_size=1, crop=128` |

### Numbers to ignore:

- Absolute loss values — 4 images is meaningless for training, the model will overfit
- Whether loss goes down consistently — with 4 samples and `shuffle=True` the ordering
  varies, so you may see occasional spikes
- PSNR — eval is disabled; even if it ran, PSNR from 10 epochs on 4 images tells you nothing

### Red flags to investigate:

| Error | Likely cause |
|---|---|
| `FileNotFoundError` on dataset paths | Path is wrong in the config |
| `ValueError: Mismatched image counts` | `low_path` and `high_path` have different numbers of images |
| `RuntimeError: Expected all tensors on same device` | A loss module was not moved to device; check `create_physics_loss` |
| `RuntimeError: MPS` + FFT error | MPS does not support all FFT ops; set `physics_losses.phase: False` to skip `PhaseEnhancedFrequencyLoss` on MPS |
| `OOM` / `CUDA out of memory` | Reduce `batch_size` to 1 and `crop_size` to 64 |
| Loss is `nan` from epoch 1 | Likely a bad `lr_initial`; try `0.00002` instead of `0.0002` |
| Phase banners never change | `physics_losses.enabled` is `False` or phase boundary keys are wrong |

---

## After the smoke test passes

Once you have seen all four phase banners, non-zero per-component losses in Phase 4,
and a checkpoint on disk, the infrastructure is confirmed working. You can then:

1. Point `low_path` / `high_path` at the real LOLBlur training split
2. Switch back to `baseline.yml` or `physics_finetune.yml` (300 epochs, full schedule)
3. Increase `batch_size` to 4, `crop_size` to 256, `n_workers` to 4
4. Enable `enhance: True` if you want the full loss suite

The smoke-test checkpoint is safe to delete — it was trained on 4 images and has no research value.
