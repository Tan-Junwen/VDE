<div align="center">
  <img src="./assets/logo.png" style="width:auto; height:120px;" alt="VDE Logo">
</div>

# [CVPR 2026] VDE: Training-Free Accelerating Rectified Flow Model via Velocity Decomposition and Estimation

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block">
    <a href="[Personal Page Link]" target="_blank">[Your Name 1]</a><sup>1,2*</sup>,
  </span>
  <span class="author-block">
    <a href="[Personal Page Link]" target="_blank">[Your Name 2]</a><sup>1</sup>,
  </span>
  <span class="author-block">
    <a href="[Personal Page Link]" target="_blank">[Your Name 3]</a><sup>2†</sup>
  </span>
</div>

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block"><sup>1</sup>[Your Institution / University 1],</span>
  <span class="author-block"><sup>2</sup>[Your Institution / University 2]</span>
</div>

<div class="is-size-5 publication-authors" align="center">
  (* Equal contribution. † Corresponding author.)
</div>

<h5 align="center">

<a href="[Project Page Link]" target="_blank">
  <img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Project Page">
</a>
<a href="[Arxiv Link]" target="_blank">
  <img src="https://img.shields.io/badge/Paper-PDF-critical.svg?logo=adobeacrobatreader" alt="Paper">
</a>
<a href="./LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License">
</a>
<a href="[GitHub Repo Link]/stargazers" target="_blank">
  <img src="https://img.shields.io/github/stars/[Your_GitHub_Username]/VDE.svg?style=social" alt="GitHub Stars">
</a>

</h5>

![VDE Overview](./assets/vde_overview.png)
> **Figure 1.** Comparison between VDE and standard 50-step sampling across Flux, Qwen-Image, and Wan2.1. VDE achieves comparable visual quality with dramatically reduced runtime (up to 3.01× speedup).

## 💡 Introduction

Though Rectified Flow (RF) models have achieved remarkable performance in visual generation, their practical deployments are challenged by slow inference speeds. Previous training-free acceleration methods (like TeaCache, EasyCache) typically follow a **caching-and-reusing** paradigm, neglecting the growing mismatch between static cached values and evolving inputs.

We propose **Velocity Decomposition and Estimation (VDE)**, a novel method that shifts the paradigm from *caching-and-reusing* to **decomposing-and-estimating**. 
- VDE decomposes the model's velocity output into components parallel and orthogonal to the input.
- It exploits the temporal predictability of the components' coefficients (strong local linearity) and the consistency of the orthogonal direction.
- VDE periodically anchors the model's state and precisely estimates subsequent outputs analytically in an inherently **input-adaptive** manner.

VDE achieves up to **2.04× - 3.22× acceleration** with minimal loss in visual quality, outperforming the best cache-based baseline (EasyCache-slow) by **19.5% in SSIM**, **30.3% in PSNR**, and reducing **LPIPS by 55.4%** in image generation.

---

## 🔥 Latest News
-[2026/03/xx] ✨ Code and demo for **VDE** are officially released! Support [Flux-dev],[Qwen-Image], and [Wan2.1].
- [2026/02/xx] 🎉 **VDE** is accepted by **CVPR 2026**! 

---

## ⚡ Performance & Demos

### 1. FLUX-dev (Text-to-Image)

**Baseline Latency (T=50): 8.20s**
| Method | Speedup | Latency | SSIM (↑) | PSNR (↑) | LPIPS (↓) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **VDE-slow** | **2.21×** | 3.70 s | **0.8877** | **25.81** | **0.1243** |
| **VDE-medium** | **2.70×** | 3.04 s | 0.8499 | 24.02 | 0.1679 |
| **VDE-fast** | **3.01×** | 2.72 s | 0.8267 | 23.19 | 0.1997 |
| *EasyCache-fast* | *2.91×* | *2.81 s* | *0.7240* | *19.59* | *0.3197* |

<details>
<summary><b>🖼️ Click to view visual comparisons on FLUX-dev</b></summary>
<div align="center">
  <img src="./assets/flux_comparison.png" alt="Flux Visual Comparison">
</div>
</details>

### 2. Qwen-Image (Text-to-Image)

**Baseline Latency (T=50): 12.53s**
| Method | Speedup | Latency | SSIM (↑) | PSNR (↑) | LPIPS (↓) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **VDE-slow** | **2.04×** | 6.14 s | **0.9362** | **28.58** | **0.0691** |
| **VDE-fast** | **2.70×** | 4.64 s | 0.8967 | 25.46 | 0.1096 |
| *TeaCache-fast* | *2.69×* | *4.66 s* | *0.5596* | *14.43* | *0.4773* |

### 3. Wan2.1-1.3B (Text-to-Video)

**Baseline Latency (T=50, 81 frames, 832×480): 175.35s**
| Method | Speedup | Latency | VBench (%) ↑ | SSIM (↑) | LPIPS (↓) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **T=50 (Original)** | 1.00× | 175.35 s | 81.30 | - | - |
| **VDE-slow** | **2.08×** | 84.18 s | **80.32** | **0.8902** | **0.0554** |
| **VDE-fast** | **2.50×** | 70.11 s | **80.43** | **0.8658** | **0.0754** |

---

## 🛠️ Supported Models

VDE currently supports and has been fully tested on the following Rectified Flow models:

**Text-to-Image**
-[x] [FLUX.1-dev](https://github.com/black-forest-labs/flux)
- [x] [Qwen-Image](https://github.com/QwenLM/Qwen-Image)

**Text-to-Video**
- [x] [Wan2.1-1.3B](https://github.com/Wan-Video/Wan2.1)

*(More models are coming soon! Contributions and PRs are highly welcome.)*

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/[Your_GitHub_Username]/VDE.git
cd VDE
conda create -n vde python=3.10
conda activate vde
pip install -r requirements.txt
