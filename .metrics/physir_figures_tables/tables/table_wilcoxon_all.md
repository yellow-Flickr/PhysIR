<!-- improvement = A - B; LPIPS sign-flipped so + = better. Rows joined on fname (stable image id). Reference impl: paired_wilcoxon.py -->
| comparison                            | metric   |   median_delta |   pct_A_better |     z |        p | sig   |   effect_r |   n |
|:--------------------------------------|:---------|---------------:|---------------:|------:|---------:|:------|-----------:|----:|
| full vs baseline @384                 | psnr     |         2.1626 |           72.2 |  5.74 | 9.48e-09 | ***   |      0.605 |  90 |
| full vs baseline @384                 | ssim     |         0.0082 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| full vs baseline @384                 | lpips    |         0.0132 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| full vs percep @384                   | psnr     |         0.592  |           70   |  1.55 | 0.121    | ns    |      0.164 |  90 |
| full vs percep @384                   | ssim     |         0.0044 |           97.8 |  8.15 | 3.52e-16 | ***   |      0.86  |  90 |
| full vs percep @384                   | lpips    |        -0.0031 |           23.3 | -4.44 | 8.99e-06 | ***   |      0.468 |  90 |
| percep vs baseline @384               | psnr     |         1.3016 |           87.8 |  7.45 | 9.06e-14 | ***   |      0.786 |  90 |
| percep vs baseline @384               | ssim     |         0.0038 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| percep vs baseline @384               | lpips    |         0.0169 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| PhysIR vs DarkIR-m released           | psnr     |         0.3613 |           61.1 |  2.1  | 0.0359   | *     |      0.221 |  90 |
| PhysIR vs DarkIR-m released           | ssim     |         0      |           51.1 | -1.77 | 0.077    | ns    |      0.186 |  90 |
| PhysIR vs DarkIR-m released           | lpips    |        -0.0085 |            0   | -8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| percep-repro vs DarkIR-m              | psnr     |         0.1444 |           56.7 |  2.07 | 0.0388   | *     |      0.218 |  90 |
| percep-repro vs DarkIR-m              | ssim     |        -0.0048 |            0   | -8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| percep-repro vs DarkIR-m              | lpips    |        -0.006  |           10   | -7.89 | 3.06e-15 | ***   |      0.832 |  90 |
| full vs baseline @256                 | psnr     |         1.7277 |           88.9 |  8.01 | 1.12e-15 | ***   |      0.845 |  90 |
| full vs baseline @256                 | ssim     |         0.0138 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| full vs baseline @256                 | lpips    |         0.0288 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| phase vs baseline @256                | psnr     |         2.0438 |           70   |  4.97 | 6.65e-07 | ***   |      0.524 |  90 |
| phase vs baseline @256                | ssim     |         0.011  |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| phase vs baseline @256                | lpips    |         0.0205 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| retinex vs baseline @256              | psnr     |         1.1181 |           87.8 |  7.63 | 2.26e-14 | ***   |      0.805 |  90 |
| retinex vs baseline @256              | ssim     |         0.0039 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| retinex vs baseline @256              | lpips    |         0.0083 |           97.8 |  8.22 | 1.99e-16 | ***   |      0.867 |  90 |
| percep vs baseline @256               | psnr     |         0.13   |           63.3 |  1.82 | 0.068    | ns    |      0.192 |  90 |
| percep vs baseline @256               | ssim     |         0.0004 |           70   |  3.97 | 7.33e-05 | ***   |      0.418 |  90 |
| percep vs baseline @256               | lpips    |         0.0133 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| blur vs baseline @256                 | psnr     |        -0.1095 |           45.6 | -0.33 | 0.743    | ns    |      0.035 |  90 |
| blur vs baseline @256                 | ssim     |        -0.0002 |           34.4 | -0.85 | 0.397    | ns    |      0.089 |  90 |
| blur vs baseline @256                 | lpips    |         0.0002 |           52.2 | -1.1  | 0.271    | ns    |      0.116 |  90 |
| full vs phase @256 (retinex marginal) | psnr     |        -0.8749 |           37.8 |  0.61 | 0.542    | ns    |      0.064 |  90 |
| full vs phase @256 (retinex marginal) | ssim     |         0.0024 |           87.8 |  6.87 | 6.4e-12  | ***   |      0.724 |  90 |
| full vs phase @256 (retinex marginal) | lpips    |         0.0056 |           95.6 |  8.1  | 5.42e-16 | ***   |      0.854 |  90 |
| full vs retinex @256 (phase marginal) | psnr     |         0.3905 |           55.6 |  3.05 | 0.00227  | **    |      0.322 |  90 |
| full vs retinex @256 (phase marginal) | ssim     |         0.0098 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| full vs retinex @256 (phase marginal) | lpips    |         0.0223 |           96.7 |  8.21 | 2.21e-16 | ***   |      0.865 |  90 |
| full vs percep @256                   | psnr     |         1.3779 |           88.9 |  7.79 | 6.82e-15 | ***   |      0.821 |  90 |
| full vs percep @256                   | ssim     |         0.0125 |          100   |  8.24 | 1.74e-16 | ***   |      0.868 |  90 |
| full vs percep @256                   | lpips    |         0.0131 |           94.4 |  8.1  | 5.6e-16  | ***   |      0.854 |  90 |
| full+percep vs full @256              | psnr     |         0.1619 |           58.9 |  0.22 | 0.823    | ns    |      0.024 |  90 |
| full+percep vs full @256              | ssim     |         0.0004 |           61.1 |  1.62 | 0.104    | ns    |      0.171 |  90 |
| full+percep vs full @256              | lpips    |        -0.001  |           42.2 | -2.47 | 0.0136   | *     |      0.26  |  90 |
| blur-instead-of-edge vs edge+blur     | psnr     |         0.2065 |           57.8 |  1.1  | 0.271    | ns    |      0.116 |  90 |
| blur-instead-of-edge vs edge+blur     | ssim     |        -0.0007 |           30   | -4.21 | 2.59e-05 | ***   |      0.443 |  90 |
| blur-instead-of-edge vs edge+blur     | lpips    |        -0.0014 |           31.1 | -1.72 | 0.0861   | ns    |      0.181 |  90 |