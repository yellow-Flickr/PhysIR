| comparison | metric | clip mean Δ | 95% CI (clip bootstrap) | clip-level p | frame-level p | clips favouring A |
|---|---|---|---|---|---|---|
| full vs baseline @384 | psnr | +0.6113 | [+0.0103, +1.2073] | ns | *** | 21/30 |
| full vs baseline @384 | ssim | +0.0175 | [+0.0132, +0.0216] | *** | *** | 29/30 |
| full vs baseline @384 | lpips | +0.0306 | [+0.0268, +0.0346] | *** | *** | 30/30 |
| full vs percep @384 | psnr | +0.3593 | [-0.2078, +0.9283] | ns | *** | 18/30 |
| full vs percep @384 | ssim | +0.0126 | [+0.0093, +0.0158] | *** | *** | 28/30 |
| full vs percep @384 | lpips | +0.0039 | [+0.0006, +0.0070] | * | *** | 19/30 |
| percep vs baseline @384 | psnr | +0.2520 | [-0.2496, +0.7486] | ns | *** | 19/30 |
| percep vs baseline @384 | ssim | +0.0049 | [+0.0026, +0.0069] | *** | *** | 27/30 |
| percep vs baseline @384 | lpips | +0.0268 | [+0.0234, +0.0301] | *** | *** | 30/30 |
| PhysIR vs DarkIR-m released | psnr | +0.2821 | [-0.3730, +0.9399] | ns | *** | 15/30 |
| PhysIR vs DarkIR-m released | ssim | +0.0009 | [-0.0018, +0.0045] | ns | * | 11/30 |
| PhysIR vs DarkIR-m released | lpips | -0.0053 | [-0.0069, -0.0037] | *** | *** | 3/30 |
| percep-repro vs DarkIR-m | psnr | -0.0772 | [-0.7083, +0.5507] | ns | ns | 15/30 |
| percep-repro vs DarkIR-m | ssim | -0.0117 | [-0.0164, -0.0060] | *** | *** | 4/30 |
| percep-repro vs DarkIR-m | lpips | -0.0092 | [-0.0116, -0.0066] | *** | *** | 4/30 |
| full vs baseline @256 | psnr | +1.5940 | [+0.9467, +2.2863] | *** | *** | 26/30 |
| full vs baseline @256 | ssim | +0.0329 | [+0.0280, +0.0378] | *** | *** | 30/30 |
| full vs baseline @256 | lpips | +0.0516 | [+0.0473, +0.0559] | *** | *** | 30/30 |
| phase vs baseline @256 | psnr | +0.9637 | [+0.4002, +1.5013] | ** | *** | 22/30 |
| phase vs baseline @256 | ssim | +0.0242 | [+0.0202, +0.0282] | *** | *** | 30/30 |
| phase vs baseline @256 | lpips | +0.0369 | [+0.0329, +0.0407] | *** | *** | 30/30 |
| retinex vs baseline @256 | psnr | +0.4835 | [+0.0840, +0.8911] | * | *** | 18/30 |
| retinex vs baseline @256 | ssim | +0.0094 | [+0.0071, +0.0119] | *** | *** | 29/30 |
| retinex vs baseline @256 | lpips | +0.0132 | [+0.0110, +0.0154] | *** | *** | 30/30 |
| percep vs baseline @256 | psnr | +0.4976 | [+0.1942, +0.7991] | * | *** | 18/30 |
| percep vs baseline @256 | ssim | +0.0026 | [+0.0009, +0.0045] | * | *** | 19/30 |
| percep vs baseline @256 | lpips | +0.0215 | [+0.0189, +0.0243] | *** | *** | 30/30 |
| blur vs baseline @256 | psnr | +0.1871 | [-0.1901, +0.5480] | ns | *** | 18/30 |
| blur vs baseline @256 | ssim | -0.0006 | [-0.0028, +0.0015] | ns | ** | 16/30 |
| blur vs baseline @256 | lpips | -0.0012 | [-0.0030, +0.0005] | ns | *** | 16/30 |
| full vs phase @256 | psnr | +0.6304 | [+0.2094, +1.0925] | * | *** | 22/30 |
| full vs phase @256 | ssim | +0.0087 | [+0.0067, +0.0108] | *** | *** | 28/30 |
| full vs phase @256 | lpips | +0.0147 | [+0.0129, +0.0166] | *** | *** | 30/30 |
| full vs retinex @256 | psnr | +1.1105 | [+0.5354, +1.7464] | *** | *** | 23/30 |
| full vs retinex @256 | ssim | +0.0235 | [+0.0196, +0.0275] | *** | *** | 30/30 |
| full vs retinex @256 | lpips | +0.0385 | [+0.0343, +0.0426] | *** | *** | 30/30 |
| full vs percep @256 | psnr | +1.0964 | [+0.4658, +1.8207] | ** | *** | 23/30 |
| full vs percep @256 | ssim | +0.0303 | [+0.0252, +0.0354] | *** | *** | 30/30 |
| full vs percep @256 | lpips | +0.0301 | [+0.0257, +0.0344] | *** | *** | 30/30 |
| full+percep vs full @256 | psnr | -0.0375 | [-0.3416, +0.2413] | ns | ns | 18/30 |
| full+percep vs full @256 | ssim | -0.0002 | [-0.0014, +0.0010] | ns | ns | 18/30 |
| full+percep vs full @256 | lpips | -0.0000 | [-0.0009, +0.0008] | ns | ns | 15/30 |
| blur-noedge vs edge+blur | psnr | +0.3292 | [-0.0416, +0.7205] | ns | *** | 16/30 |
| blur-noedge vs edge+blur | ssim | +0.0023 | [+0.0001, +0.0050] | ns | *** | 18/30 |
| blur-noedge vs edge+blur | lpips | +0.0005 | [-0.0009, +0.0022] | ns | ns | 13/30 |
