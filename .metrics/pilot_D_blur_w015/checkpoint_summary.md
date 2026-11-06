# Checkpoint summary — pilot_D_blur_w015

Started: 2026-06-29 16:01


## Epoch 5  ⭐ **BEST**
_Saved: 2026-06-30 01:00_

| Metric | Value |
|--------|-------|
| Phase | 3 (+BlurGrad) |
| val PSNR  | 23.2330 dB |
| val SSIM  | 0.7384 |
| val LPIPS | 0.4211 |
| blur σ | 1.7058 |
| loss_total | 0.035823 |
| loss_base | 0.030789 |
| loss_retinex | 0.000000 |
| loss_blur | 0.005034 |
| loss_phase_freq | 0.000000 |

**Loss balance:**
```
  total         :   0.0358 ( 50.0%) █████████████████████████
  base          :   0.0308 ( 43.0%) █████████████████████
  retinex       :   0.0000 (  0.0%) 
  blur          :   0.0050 (  7.0%) ███
  phase_freq    :   0.0000 (  0.0%) 
  TOTAL         :   0.0716
```

### Perceptual eval — epoch 10
| Metric | Score | Direction |
|--------|------:|-----------|
| NIQE  | 5.6876 | ↓ lower = better |
| NRQM  | 5.0215 | ↑ higher = better |
| MUSIQ | 33.7501 | ↑ higher = better |

## Epoch 10  ⭐ **BEST**
_Saved: 2026-06-30 02:11_

| Metric | Value |
|--------|-------|
| Phase | 3 (+BlurGrad) |
| val PSNR  | 23.4680 dB |
| val SSIM  | 0.7555 |
| val LPIPS | 0.3982 |
| blur σ | 1.9387 |
| loss_total | 0.030120 |
| loss_base | 0.024928 |
| loss_retinex | 0.000000 |
| loss_blur | 0.005191 |
| loss_phase_freq | 0.000000 |

**Loss balance:**
```
  total         :   0.0301 ( 50.0%) █████████████████████████
  base          :   0.0249 ( 41.4%) ████████████████████
  retinex       :   0.0000 (  0.0%) 
  blur          :   0.0052 (  8.6%) ████
  phase_freq    :   0.0000 (  0.0%) 
  TOTAL         :   0.0602
```

## Epoch 15  ⭐ **BEST**
_Saved: 2026-06-30 02:58_

| Metric | Value |
|--------|-------|
| Phase | 3 (+BlurGrad) |
| val PSNR  | 23.7470 dB |
| val SSIM  | 0.7612 |
| val LPIPS | 0.3868 |
| blur σ | 2.1273 |
| loss_total | 0.025946 |
| loss_base | 0.020667 |
| loss_retinex | 0.000000 |
| loss_blur | 0.005278 |
| loss_phase_freq | 0.000000 |

**Loss balance:**
```
  total         :   0.0259 ( 50.0%) ████████████████████████
  base          :   0.0207 ( 39.8%) ███████████████████
  retinex       :   0.0000 (  0.0%) 
  blur          :   0.0053 ( 10.2%) █████
  phase_freq    :   0.0000 (  0.0%) 
  TOTAL         :   0.0519
```

## Epoch 20  ⭐ **BEST**
_Saved: 2026-06-30 03:45_

| Metric | Value |
|--------|-------|
| Phase | 3 (+BlurGrad) |
| val PSNR  | 23.7922 dB |
| val SSIM  | 0.7683 |
| val LPIPS | 0.3798 |
| blur σ | 2.1871 |
| loss_total | 0.023691 |
| loss_base | 0.018337 |
| loss_retinex | 0.000000 |
| loss_blur | 0.005354 |
| loss_phase_freq | 0.000000 |

**Loss balance:**
```
  total         :   0.0237 ( 50.0%) █████████████████████████
  base          :   0.0183 ( 38.7%) ███████████████████
  retinex       :   0.0000 (  0.0%) 
  blur          :   0.0054 ( 11.3%) █████
  phase_freq    :   0.0000 (  0.0%) 
  TOTAL         :   0.0474
```

### Perceptual eval — epoch 25
| Metric | Score | Direction |
|--------|------:|-----------|
| NIQE  | 6.0011 | ↓ lower = better |
| NRQM  | 4.8267 | ↑ higher = better |
| MUSIQ | 34.6489 | ↑ higher = better |

## Epoch 25  ⭐ **BEST**
_Saved: 2026-06-30 05:41_

| Metric | Value |
|--------|-------|
| Phase | 3 (+BlurGrad) |
| val PSNR  | 23.6651 dB |
| val SSIM  | 0.7679 |
| val LPIPS | 0.3792 |
| blur σ | 1.5620 |
| loss_total | 0.022849 |
| loss_base | 0.017532 |
| loss_retinex | 0.000000 |
| loss_blur | 0.005317 |
| loss_phase_freq | 0.000000 |

**Loss balance:**
```
  total         :   0.0228 ( 50.0%) ████████████████████████
  base          :   0.0175 ( 38.4%) ███████████████████
  retinex       :   0.0000 (  0.0%) 
  blur          :   0.0053 ( 11.6%) █████
  phase_freq    :   0.0000 (  0.0%) 
  TOTAL         :   0.0457
```

---
## Run complete
_Finished: 2026-06-30 05:41_

| | |
|---|---|
| Best epoch | 25 |
| Best val PSNR  | 23.6651 dB |
| Best val SSIM  | 0.7679 |
| Best val LPIPS | 0.3792 |
