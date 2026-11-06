<!-- LPIPS column = [-1,1] scaling (lpips spec); lpips_darkir_conv = unscaled [0,1], their convention -->
| model                |    psnr |   ssim |   lpips |    n |   lpips_darkir_conv |
|:---------------------|--------:|-------:|--------:|-----:|--------------------:|
| PhysIR (full)        | 27.2858 | 0.8838 |  0.2069 | 1800 |              0.1664 |
| baseline             | 26.6745 | 0.8663 |  0.2376 | 1800 |              0.1962 |
| abl_percep           | 26.9265 | 0.8713 |  0.2108 | 1800 |              0.1743 |
| PhysIR (re-eval)     | 27.2858 | 0.8838 |  0.2069 | 1800 |              0.1664 |
| abl_percep (re-eval) | 26.9265 | 0.8713 |  0.2108 | 1800 |              0.1743 |
| DarkIR-m released    | 27.0037 | 0.8829 |  0.2016 | 1800 |              0.162  |