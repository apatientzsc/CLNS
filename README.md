# Camera-aware Label Noise Suppression (CLNS)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**Official implementation of "Camera-aware Label Noise Suppression"**


> **Abstract:** Unsupervised visible-infrared person re-identification (US-VI-ReID) retrieves pedes-
> trian images across modalities without manual annotations. To address camera-specific
> biases that fragment identities and amplify label noise, we propose the Camera-aware
> Label Noise Suppression (CLNS) framework, a coarse-to-fine pipeline that progres-
> sively purifies noise. Specifically, the Camera-aware Prototype Calibration (CPC)
> module exploits cross-camera consistency to rectify structural errors and construct re-
> liable prototypes. Building on this, Optimal Transport Prototype Matching (OTPM)
> achieves global cross-modality alignment. At the instance level, Neighbor-guided
> Camera-domain Learning (NCL) densifies feature distributions using soft supervision,
> while a Noise-aware Memory Updating (NMU) strategy prevents error accumulation.
> On the SYSU-MM01, RegDB, and LLCM datasets, CLNS achieves Rank-1 (mAP)
> accuracies of 69.5% (65.6%), 94.8% (91.8%), and 54.6% (57.8%), respectively, significantly surpassing state-of-the-art methods.

---
- ## 📥Test Code

  Our work is based on CEIL. You can get test code from this link. The datasets and environment are same. All code will released soon.

  (https://github.com/maybeextra/CEIL)

  

---
## 🎯 Performance

Our CEIL method achieves state-of-the-art performance on three popular cross-modal Re-ID benchmarks. **(R)** indicates results with re-ranking.

### Results on RegDB

| Mode      | Rank-1 | mAP   | mINP  |
|:----------|:------:|:-----:|:-----:|
| VIS to IR | 94.79% | 91.83% | 85.8% |
| IR to VIS | 95.33% | 91.79% | 84.25% |

### Results on SYSU-MM01

| Mode          | Rank-1 | mAP   | mINP  |
|:--------------|:------:|:-----:|:-----:|
| All Search    | 69.49% | 65.64% | 51.2% |
| Indoor Search | 74.67% | 78.95% | 75.31% |

### Results on LLCM

| Mode      | Rank-1 | mAP   | mINP  |
|:----------|:------:|:-----:|:-----:|
| VIS to IR | 54.6% | 57.8% | 51.9% |
| IR to VIS | 48.8% | 52.6% | 48.8% |

---
## 📥Weights and Logs

You can download our trained models and logsfrom the links below. 

[📥 Download from GoogleDrive](https://drive.google.com/file/d/1awkIVqgyloiL18O2b2YwIYXkAaHffT85/view?usp=drive_link)

