# DarkIR — Local Training Guide

Physics-guided loss study on DarkIR-m, trained **from scratch**.

---

## Why from scratch (not fine-tuning)

The thesis claim is: *"physics-guided loss functions improve low-light image restoration."*
To support that claim the comparison must be:

```
DarkIR-m  trained from scratch  +  original losses           ← baseline
DarkIR-m  trained from scratch  +  original + physics losses  ← proposed
```

Fine-tuning from the pretrained checkpoint confounds extra training epochs with the effect of the new losses — a thesis examiner will flag it immediately. Both runs must start from random initialisation, train for the same number of epochs with identical data and optimiser settings, and differ **only** in which losses are active.

---

## Experimental design

### Required training runs

| Config | Physics losses | Purpose |
|---|---|---|
| `baseline.yml` | disabled | Replicate DarkIR-m published result (~27.00 dB on LOLBlur) |
| `physics_finetune.yml` | progressive (phases 2–3) | Main proposed result |

Both use **DarkIR-m (width=32, 3.3 M params)** — ~12 hrs per run on a Kaggle T4.

### Ablation table (from phase checkpoints)

Because `physics_finetune.yml` uses a progressive schedule, the checkpoints at phase boundaries give the ablation rows for free:

| Variant | L_retinex | L_blur | L_phase | Checkpoint |
|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | `DarkIR_m_baseline_best.pt` |
| + Retinex only | ✓ | ✗ | ✗ | ep 200 of physics run |
| + All three (full) | ✓ | ✓ | ✓ | `DarkIR_m_physics_best.pt` |

### Target result table

| Method | LOLBlur PSNR | LOLv2-real PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| DarkIR-m (published) | 27.00 | 23.87 | — | — |
| DarkIR-m (replicated, our baseline) | ~27.0x | — | — | — |
| **DarkIR-m + physics losses (ours)** | **?** | **?** | **?** | **?** |

The replication row serves as a sanity check on your training setup before committing to the full proposed run.

---

## Infrastructure overview

```
DarkIR/
├── train.py                            # main training entry-point (single device)
├── testing.py                          # evaluation on benchmark datasets
├── inference.py                        # single-image / folder inference
│
├── archs/
│   ├── DarkIR.py                       # model architecture
│   └── __init__.py                     # create_model, resume_model, save_checkpoint
│
├── losses/
│   ├── loss.py                         # all loss classes (incl. 3 physics losses)
│   └── __init__.py                     # create_loss, create_physics_loss, calculate_loss
│
├── data/
│   ├── __init__.py                     # create_train_data, create_test_data
│   └── dataset_reader/
│       ├── datapipeline.py             # MyDataset_Crop, RandomCropSame
│       ├── dataset_train.py            # generic paired training loader
│       └── dataset_*.py               # per-benchmark test loaders
│
├── options/
│   ├── train/
│   │   ├── baseline.yml               # ← baseline run (no physics losses)
│   │   └── physics_finetune.yml       # ← proposed run (physics losses, from scratch)
│   ├── test/                          # evaluation configs
│   └── inference/                     # inference configs
│
├── models/
│   ├── DarkIR_m_baseline_latest.pt    # written every save_freq epochs (baseline run)
│   ├── DarkIR_m_physics_latest.pt     # written every save_freq epochs (proposed run)
│   └── bests/
│       ├── DarkIR_m_baseline_best.pt  # best val-PSNR baseline checkpoint
│       └── DarkIR_m_physics_best.pt   # best val-PSNR proposed checkpoint
│
└── utils/
    ├── test_utils.py                   # DDP eval helpers, setup/cleanup
    └── utils.py                        # wandb helpers, path utilities
```

---

## Physics-guided losses

Three training-only loss classes in [`losses/loss.py`](losses/loss.py):

| Class | Physical prior | Extra trainable params |
|---|---|---|
| `RetinexLoss` | `I = R ⊙ L` — illumination smoothness + reconstruction consistency | none |
| `BlurAwareGradientLoss` | `y = x ⊗ k` — re-blur gradient consistency | **learnable σ** (PSF sigma, updated by backprop) |
| `PhaseEnhancedFrequencyLoss` | Fourier phase carries structural information | none |

Instantiated by `create_physics_loss()` in [`losses/__init__.py`](losses/__init__.py).
The learnable σ from `BlurAwareGradientLoss` is automatically added to the optimizer.

### Progressive schedule in `physics_finetune.yml`

| Phase | Epochs | Active losses | Purpose |
|---|---|---|---|
| 1 | 1 – 100 | L1 + Edge + Enhance | Warm-up (identical to baseline) |
| 2 | 101 – 200 | + RetinexLoss | Ablation checkpoint at ep 200 |
| 3 | 201 – 300 | + BlurGrad + PhaseFreq | Full proposed method |

---

## Step-by-step setup

### 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon (MPS): PyTorch ≥ 2.0 includes MPS support — no extra steps.

### 2. Prepare your dataset

The training loader expects two flat directories with **sorted, matching filenames**:

```
LOLBlur/
├── train/
│   ├── low_blur_noise/     # low-light images:  00001.png, 00002.png, …
│   └── high_sharp_scaled/  # ground-truth images (same filenames)
└── test/
    ├── low_blur_noise/
    └── high_sharp_scaled/
```

LOLBlur is the recommended training set (~3 800 pairs, best match for DarkIR's published training). Download from the [LOL-Blur paper repo](https://github.com/lingyzhu0101/low-light-image-enhancement-and-deblurring).

### 3. Edit the dataset paths

Both configs need updating before the first run. Set the same paths in both:

**`options/train/baseline.yml`** and **`options/train/physics_finetune.yml`**:

```yaml
datasets:
  train:
    low_path:  /absolute/path/to/LOLBlur/train/low_blur_noise
    high_path: /absolute/path/to/LOLBlur/train/high_sharp_scaled
  val:
    test_path: /absolute/path/to/data/datasets   # parent of LOLBlur, LOL-v2, etc.
```

### 4. Run the baseline

```bash
python train.py -p options/train/baseline.yml
```

Let this run to completion (300 epochs). If the best val PSNR lands within ~0.1 dB of the published 27.00 dB your training setup is correct.

### 5. Run the proposed model

```bash
python train.py -p options/train/physics_finetune.yml
```

Both configs start from **random initialisation** (`save.pretrained: null`). The script auto-selects device (CUDA → MPS → CPU).

To resume an interrupted run, re-run the same command — the script resumes from `save.path` if it exists.

### 6. Evaluate

```bash
# Evaluate a checkpoint on all LOL benchmarks
python testing.py -p options/test/AllLOL.yml
```

Edit `save.path` in the test config to point at whichever checkpoint you want.

---

## Kaggle strategy (if running on T4)

Each run is ~12 hrs on a single T4. Kaggle sessions are 9 hrs.

1. Start the run with `save_freq: 10` — a checkpoint is written every 10 epochs.
2. After the session ends, download the `*_latest.pt` checkpoint.
3. Upload it to the next session and place it at the same `save.path`.
4. Re-run `python train.py -p ...` — it resumes from epoch N automatically.

Typically takes 2 sessions per run (~$0 on free tier with GPU quota).

---

## Checkpoint format

All checkpoints written by `train.py`:

```python
{
  'epoch'     : int,
  'model'     : state_dict,        # for resuming training
  'params'    : state_dict,        # identical key — compatible with inference.py
  'optimizer' : optim state_dict,
  'scheduler' : scheduler state_dict,
  'blur_sigma': float (optional),  # learned PSF sigma at save time
}
```

The `'params'` key is duplicated so checkpoints load directly into `inference.py` and the Kaggle notebooks without conversion.

---

## Wandb logging (optional)

Set `wandb.init: True` in the config. Per-epoch metrics logged automatically:
- `total`, `base`, `retinex`, `blur`, `phase_freq` losses
- `val_psnr`, `lr`, `blur_sigma`

Give each run a distinct `wandb.name` (`DarkIR-m_baseline_scratch` vs `DarkIR-m_physics_scratch`) so you can overlay the training curves.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError` on dataset paths | Check `datasets.train.{low_path,high_path}` in the config |
| `ValueError: Mismatched image counts` | Both dirs must have the same number of files, sorted to the same order |
| OOM | Reduce `datasets.train.batch_size` (4 → 2) or `crop_size` (256 → 192) |
| Validation loader fails | Check `datasets.val.test_path`; training continues without val if it fails |
| `VGGLoss` fails on CPU | Downloads VGG19 weights from torchvision on first use — needs internet |
| MPS slow on Phase 3 | MPS FFT is limited; set `physics_losses.phase: False` to skip `PhaseEnhancedFrequencyLoss` |
| Baseline PSNR far from 27.00 dB | Check LR and `enhance` loss are enabled; compare `train.lr_initial` against original DarkIR paper |
