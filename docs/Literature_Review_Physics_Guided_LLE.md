# Literature Review: Physics-Guided Approaches for Low-Light Image Restoration

*Supporting the proposal: "Physics-Guided Loss Functions for Low-Light Image Restoration"*

---

## 1. Introduction

Low-light image restoration addresses degradations arising from dim environments, long exposure, and camera shake. Recent deep learning methods achieve strong results but often treat restoration as a black-box mapping, with limited use of physical image formation models. This review covers prior work at the intersection of physics-based models and deep learning for low-light enhancement, deblurring, and joint restoration, and positions the proposed physics-guided loss functions within this landscape.

---

## 2. Retinex Theory and Illumination-Reflectance Decomposition

### 2.1 Classical Retinex

Retinex theory, developed by Land [11], models perceived intensity as the product of reflectance and illumination:

$$I = R \odot L$$

where $R$ encodes scene reflectance (content) and $L$ encodes illumination (lighting). Reflectance is assumed relatively constant under illumination changes; illumination varies smoothly in space.

### 2.2 Deep Retinex Decomposition

**Wei et al.** [1] introduced Retinex-Net, which learns illumination-reflectance decomposition without ground-truth decompositions. Decom-Net and Enhance-Net are trained with:

- **Reflectance consistency**: Paired low/normal-light images share reflectance.
- **Illumination smoothness**: Spatial regularization (e.g., total variation) on illumination maps.

This work established that differentiable Retinex constraints enable end-to-end training and has influenced many subsequent methods.

**URetinex-Net** [2] uses a deep unfolding framework with learnable modules to adaptively fit implicit Retinex priors. It includes optimization-inspired blocks for noise suppression and detail preservation.

**Lecert et al.** [18] propose new regularizations for Retinex decomposition of low-light images, showing that alternative smoothness and consistency terms can improve stability.

### 2.3 Retinex in Modern Architectures

**RetinexFormer** [12] uses a one-stage Retinex-based transformer with an Illumination-Guided Transformer (IGT), achieving strong results on multiple benchmarks. The method shows that illumination-aware design can improve both quality and generalization.

**DI-Retinex** [19] revisits Retinex in the context of digital imaging, introducing an image-adaptive masked reverse degradation loss in Gamma space and a variance suppression loss to handle noise and quantization.

### 2.4 Gap Addressed by the Proposal

Prior methods typically use Retinex either through dedicated decomposition networks or architectural biases. The proposed approach applies Retinex constraints as **loss terms** on the final (and optionally intermediate) outputs of a joint enhancement–deblurring network, without adding decomposition sub-networks.

---

## 3. Blur Formation and Deblurring

### 3.1 Physical Model

Blur is commonly modeled as convolution with a point spread function (PSF):

$$y = x \otimes k$$

where $y$ is the blurred observation, $x$ is the sharp image, and $k$ is the blur kernel. In low-light conditions, long exposure increases motion blur severity.

### 3.2 Learning-Based Deblurring

Most recent deblurring methods use data-driven learning rather than explicit kernel estimation. **Kim and Lee** [3] and similar works use segmentation and kernel estimation for dynamic scenes.

**Sharpness-based losses** [14] show that training with perceptual or sharpness-oriented losses can outperform pixel-wise losses (MAE, MSE) for deblurring, improving LPIPS and sharpness metrics.

### 3.3 Physics-Informed Blur Learning

Recent work embeds blur physics into learning. **Physics-informed blur learning** frameworks [13] use optical aberration models and MLP-based PSF estimation. **CircleFlow** and related methods estimate anisotropic PSFs via flow-guided localization and implicit representations.

**Blur kernel space** methods encode blur operators in a differentiable space, enabling networks to generalize to unseen kernel types.

### 3.4 Gap Addressed by the Proposal

Existing joint enhancement–deblurring methods (e.g., LEDNet) do not explicitly enforce consistency with the blur formation model. The proposed blur-aware gradient loss enforces:

$$\|\nabla(x_{\text{pred}} \otimes k_{\text{est}}) - \nabla y\|^2$$

where $k_{\text{est}}$ is estimated parametrically (Gaussian) or via FFT, linking the sharp output to the input’s gradient structure under re-blurring.

---

## 4. Frequency Domain and Phase in Image Restoration

### 4.1 Fourier Structure of Images

In the Fourier domain, **amplitude** typically encodes overall intensity and coarse structure, while **phase** encodes edges and fine detail. Oppenheim and Lim showed that phase often carries more perceptual information than amplitude.

### 4.2 Frequency-Domain Losses

**Focal Frequency Loss** [4] proposes adaptive weighting of frequency components to address spectral bias in generative models. It focuses on amplitude-based frequency distances and has been applied to VAE, pix2pix, SPADE, and StyleGAN2.

**Guided Frequency Loss** [15] combines Charbonnier, Laplacian pyramid, and gradual frequency terms to balance spatial and frequency learning in super-resolution and denoising.

### 4.3 Phase in Low-Light and Deblurring

**FourierDiff** [5] exploits the observation that luminance concentrates in amplitude and structure in phase. It uses Fourier priors in a diffusion model for zero-shot joint low-light enhancement and deblurring, with separate amplitude and phase refinement stages.

**NeuralRemaster** and related work use phase-preserving diffusion to maintain structural consistency.

### 4.4 Gap Addressed by the Proposal

DarkIR’s FreMLP operates on Fourier amplitude only. The proposed phase-enhanced frequency loss adds phase supervision (via cos/sin or phase gradients) to prevent structure loss during illumination adjustment, extending amplitude-only frequency losses used in existing architectures.

---

## 5. Joint Low-Light Enhancement and Deblurring

### 5.1 Task Coupling

In night photography, low light and motion blur co-occur. Cascading separate enhancement and deblurring networks can propagate errors and produce suboptimal results. Joint methods aim to exploit synergy between the two tasks.

### 5.2 LEDNet and LOL-Blur

**Zhou et al.** [9] introduced LEDNet and the **LOL-Blur** dataset, the first large-scale benchmark for joint low-light enhancement and deblurring. LOL-Blur has 10,200 training and 1,800 test pairs with varied darkness and blur. LEDNet uses a network designed for joint processing.

### 5.3 Recent Joint Methods

**JUDE** [10] uses deep joint unrolling inspired by physical models, incorporating Retinex and blur models for iterative deblurring and decomposition into sharp reflectance and illumination.

**FourierDiff** [5] applies Fourier priors in a pre-trained diffusion model for zero-shot joint enhancement and deblurring without paired data.

**DEvUDP** proposes an unsupervised approach for individual or joint enhancement and deblurring using transformation and self-regression branches.

### 5.4 DarkIR as Baseline

**Feijoo et al.** [16] present DarkIR, an efficient CNN for joint low-light restoration. It uses an asymmetric encoder–decoder with FreMLP (frequency MLP) for illumination and dilated spatial attention for deblurring. DarkIR achieves state-of-the-art results on LOLBlur, LOLv2, and Real-LOLBlur.

---

## 6. Physics-Guided and Physics-Informed Deep Learning

### 6.1 Physics-Guided Losses

Several works use physical priors to design losses or regularizers:

**Physics-guided noise neural proxy** [6] uses physics-guided noise decoupling (PND), physics-guided proxy models (PPM), and differentiable distribution loss (DDL) for low-light RAW denoising.

**PiCat** [7] uses physics-informed color-aware transforms with image decomposition for color consistency in low-light enhancement.

**Rectified flow** [8] combines physics-based noise synthesis with rectified flow for low-light RAW enhancement.

### 6.2 Comparison to the Proposal

These methods often change architecture, sampling, or data pipelines. The proposal instead keeps the architecture fixed and adds **physics-guided loss terms**, offering a lighter-weight and more modular approach.

---

## 7. Perceptual and Quality Metrics

**LPIPS** [17] uses deep features to measure perceptual similarity and is widely adopted in low-light restoration. The proposal retains LPIPS while adding physics-based terms.

**PSNR and SSIM** remain standard for distortion-focused evaluation. The proposal uses both for comprehensive assessment.

---

## 8. Synthesis and Positioning

The literature shows that:

1. **Retinex** constraints (smoothness, consistency) are established in low-light enhancement but usually implemented via decomposition networks or architectural design.
2. **Blur-aware** constraints (re-blur consistency, gradient matching) appear in deblurring but have not been integrated into joint enhancement–deblurring loss design.
3. **Phase** in the Fourier domain is recognized as important (e.g., FourierDiff) but rarely used as an explicit loss term in supervised restoration.

The proposed physics-guided loss suite is the first to combine:

- Retinex decomposition loss (without a decomposition network)
- Blur-aware gradient loss (with implicit PSF estimation)
- Phase-enhanced frequency loss (extending amplitude-only supervision)

within a single training objective for low-light image restoration. This positions the work as a principled bridge between classical image formation models and modern deep learning, with a focus on interpretable, plug-in loss design.

---

## 9. References

[1] C. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement," *BMVC*, 2018.  
[2] W. Wu et al., "URetinex-Net: Retinex-based Deep Unfolding Network for Low-Light Image Enhancement," *CVPR*, 2022.  
[3] T. H. Kim and K. M. Lee, "Segmentation-free Dynamic Scene Deblurring," *CVPR*, 2014.  
[4] L. Jiang, B. Dai, W. Wu, and C. C. Loy, "Focal Frequency Loss for Image Reconstruction and Synthesis," *ICCV*, 2021.  
[5] Z. Lv, F. Wang, and H. Lu, "Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light Enhancement and Deblurring," *CVPR*, 2024.  
[6] Y. Zhang et al., "Physics-guided Noise Neural Proxy for Practical Low-light Raw Image Denoising," *arXiv*, 2023.  
[7] X. Liu et al., "Learning Physics-Informed Color-Aware Transforms for Low-Light Image Enhancement," *arXiv*, 2025.  
[8] Y. Chen et al., "Physics-Guided Rectified Flow for Low-light RAW Image Enhancement," *arXiv*, 2025.  
[9] S. Zhou et al., "LEDNet: Joint Low-Light Enhancement and Deblurring in the Dark," *ECCV*, 2022.  
[10] JUDE, "Deep Joint Unrolling for Deblurring and Low-Light Image Enhancement," 2025.  
[11] E. H. Land, "The Retinex Theory of Color Vision," *Scientific American*, 1977.  
[12] Y. Cai et al., "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement," *ICCV*, 2023.  
[13] Physics-Informed Blur Learning, *CVPR*, 2025.  
[14] A. López et al., "A Sharpness Based Loss Function for Removing Out-of-Focus Blur," *arXiv*, 2024.  
[15] Guided Frequency Loss for Image Restoration.  
[16] D. Feijoo et al., "DarkIR: Robust Low-Light Image Restoration," *CVPR*, 2025.  
[17] R. Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," *CVPR*, 2018.  
[18] P. Lecert, A. Bugeau, and G. Facciolo, "A New Regularization for Retinex Decomposition of Low-Light Images," *HAL*, 2022.  
[19] DI-Retinex, "Digital-Imaging Retinex Theory for Low-Light Image Enhancement," *arXiv*, 2024.
